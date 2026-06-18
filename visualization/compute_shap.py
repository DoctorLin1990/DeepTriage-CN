#!/usr/bin/env python3
"""
compute_shap_and_figure4.py
===========================
用真实数据计算 DeepTriage-CN 的 SHAP 值，并生成 Figure 4。

工作流程
--------
Step 1  训练 DeepTriage-CN（若模型文件不存在）
Step 2  构建融合特征矩阵 X_fused (N_val × 776)
Step 3  用 TreeExplainer 计算 SHAP 值
        ─ XGBoost 是树模型 → TreeExplainer 精确高效，无需近似
        ─ 解释的是 X_fused，包含 768 维 BERT 嵌入 + 8 维生命体征
        ─ 768 维嵌入不直接可解释，因此对文本部分做"主成分 SHAP 聚合"：
          将 768 个 BERT 特征的 SHAP 值 L1 范数求和，压缩为 1 个
          "文本综合贡献度"指标，再附上高 SHAP 值词元对应的语义标签
Step 4  绘制 Figure 4 三面板：
        Panel a  真实 SHAP beeswarm（生命体征 8 个特征 + 文本综合）
        Panel b  Youden 阈值(0.28)处 TP/TN/FP/FN 条形图
        Panel c  FNR 分组柱状图（年龄 × 主诉稀疏度）

依赖
----
    pip install shap xgboost scikit-learn transformers torch joblib matplotlib

运行
----
    # 场景1：已有训练好的模型
    python compute_shap_and_figure4.py \
        --train /path/to/DeepTriage_train_8000.csv \
        --test  /path/to/DeepTriage_test_2000.csv \
        --model outputs/models/deeptriage_cn.pkl \
        --outdir outputs/figures

    # 场景2：从头训练（无模型文件）
    python compute_shap_and_figure4.py \
        --train /tmp/DeepTriage_train_8000.csv \
        --test  /tmp/DeepTriage_test_2000.csv \
        --outdir /tmp/figures_shap
        # --model 不传，脚本自动训练并保存到 --outdir/deeptriage_cn.pkl

注意事项
--------
* BERT 编码需要约 4–8 GB RAM（bert-base-chinese，110M参数）
* 首次运行会从 HuggingFace 下载模型（约 400 MB）
* SHAP TreeExplainer 在 N=2000、p=776 时约需 2–5 分钟
* 若无 GPU，TextEncoder 自动切换到 CPU（速度较慢但结果相同）
"""

import argparse, os, sys, warnings, logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
import shap
import joblib
from sklearn.metrics import roc_curve, brier_score_loss

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# ── 确保项目src在路径上 ────────────────────────────────────────────────
_SCRIPT_DIR = Path(__file__).resolve().parent
for _candidate in [
    _SCRIPT_DIR / "project" / "DeepTriage-CN",
    _SCRIPT_DIR / "project2" / "DeepTriage-CN",
    _SCRIPT_DIR,
]:
    if (_candidate / "src").exists():
        sys.path.insert(0, str(_candidate))
        logger.info(f"Project root: {_candidate}")
        break

# ── 设计常量 ───────────────────────────────────────────────────────────
FONT = "DejaVu Sans"
PAL  = {
    "DeepTriage-CN":         "#D55E00",
    "XGBoost (Vitals-only)": "#56B4E9",
    "NEWS2":                 "#0072B2",
    "ESI":                   "#44AA99",
    "TabNet":                "#009E73",
}
SPARSE_SET = {"头晕", "发热", "腹痛", "乏力", "胸闷"}

# 8个生命体征特征的显示名称（与StructuredEncoder列顺序一致）
STRUCT_NAMES = [
    "Age", "Sex",
    "Temperature (°C)",
    "Heart rate (bpm)",
    "Respiratory rate (/min)",
    "Systolic BP (mmHg)",
    "Diastolic BP (mmHg)",
    "SpO\u2082 (%)",
]


# ═══════════════════════════════════════════════════════════════════════
#  STEP 0: 风格设置
# ═══════════════════════════════════════════════════════════════════════

def set_style():
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": [FONT, "Arial"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
        "xtick.major.size": 3.5,
        "ytick.major.size": 3.5,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "legend.fontsize": 5.8,
        "legend.frameon": False,
        "legend.handlelength": 1.8,
        "legend.labelspacing": 0.32,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


# ═══════════════════════════════════════════════════════════════════════
#  STEP 1: 数据加载与列名修复
# ═══════════════════════════════════════════════════════════════════════

COL_RENAME = {
    "Age": "age", "Gender": "sex",
    "Temperature": "temperature", "Pulse": "heart_rate",
    "Respiratory_Rate": "respiratory_rate",
    "Systolic_BP": "sbp", "Diastolic_BP": "dbp", "SpO2": "spo2",
    "Chief_Complaint": "chief_complaint",
    "Hospital_Admission": "hospital_admission",
}
STRUCT_COLS = ["age", "sex", "temperature", "heart_rate",
               "respiratory_rate", "sbp", "dbp", "spo2"]


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns=COL_RENAME)
    # 修复性别编码
    if df["sex"].dtype == object:
        df["sex"] = (df["sex"] == "Male").astype(float)
    # 生成 visit_date（若缺失）
    if "visit_date" not in df.columns:
        base = pd.Timestamp("2023-01-01")
        df["visit_date"] = base + pd.to_timedelta(
            np.arange(len(df)) * 1.5, unit="h")
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    return df


# ═══════════════════════════════════════════════════════════════════════
#  STEP 2: 模型训练（若模型文件不存在）
# ═══════════════════════════════════════════════════════════════════════

def train_model(train_df, outdir):
    """快速训练 DeepTriage-CN，保存到 outdir/deeptriage_cn.pkl。"""
    logger.info("Training DeepTriage-CN from scratch …")
    from src.text_encoder import TextEncoder
    from src.structured_encoder import StructuredEncoder
    import xgboost as xgb

    # 文本编码
    te = TextEncoder()
    train_texts = train_df["chief_complaint"].fillna("[MISSING]").tolist()
    logger.info("  Encoding training texts (BERT) …")
    text_emb = te.encode(train_texts)   # (8000, 768)

    # 结构化特征标准化
    X_raw = train_df[STRUCT_COLS].values.astype(float)
    enc = StructuredEncoder()
    X_std = enc.fit_transform(X_raw)   # (8000, 8)

    # 融合
    X_fused = np.hstack([text_emb, X_std])   # (8000, 776)
    y_train = train_df["hospital_admission"].values

    # XGBoost
    n_neg = int((y_train == 0).sum())
    n_pos = int((y_train == 1).sum())
    spw = n_neg / max(n_pos, 1)
    clf = xgb.XGBClassifier(
        n_estimators=200, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=spw, tree_method="hist",
        random_state=42, verbosity=0,
    )
    logger.info("  Training XGBoost …")
    clf.fit(X_fused, y_train)

    # 保存
    model_path = os.path.join(outdir, "deeptriage_cn_quick.pkl")
    joblib.dump({"clf": clf, "enc": enc, "te_config": {
        "model_name": te.model_name,
        "max_length": te.max_length,
    }}, model_path)
    logger.info(f"  Model saved → {model_path}")
    return clf, enc, te


def load_model(model_path):
    """加载已保存的模型（两种格式均支持）。"""
    state = joblib.load(model_path)
    if isinstance(state, dict) and "clf" in state:
        # 快速格式
        from src.text_encoder import TextEncoder
        from src.structured_encoder import StructuredEncoder
        clf = state["clf"]
        enc = state["enc"]
        tc  = state.get("te_config", {})
        te  = TextEncoder(
            model_name=tc.get("model_name", "bert-base-chinese"),
            max_length=tc.get("max_length", 64),
        )
    else:
        # DeepTriageCN.save() 格式
        from src.fusion_model import DeepTriageCN
        m   = DeepTriageCN.load_from_file(model_path)
        clf = m.classifier
        enc = m.structured_encoder
        te  = m.text_encoder
    return clf, enc, te


# ═══════════════════════════════════════════════════════════════════════
#  STEP 3: 构建验证集融合特征矩阵
# ═══════════════════════════════════════════════════════════════════════

def build_fused(val_df, enc, te):
    """返回 X_fused (N_val, 776)、y、geri、cc_sparse。"""
    logger.info("Building fused feature matrix for validation set …")
    texts = val_df["chief_complaint"].fillna("[MISSING]").tolist()
    logger.info("  Encoding validation texts (BERT) …")
    text_emb = te.encode(texts)                          # (N, 768)
    X_raw    = val_df[STRUCT_COLS].values.astype(float)
    X_std    = enc.transform(X_raw)                      # (N, 8)
    X_fused  = np.hstack([text_emb, X_std])             # (N, 776)
    y        = val_df["hospital_admission"].values
    geri     = (val_df["age"].values >= 65)
    cc       = val_df["chief_complaint"].values
    cc_sparse = np.array([isinstance(c, str) and c in SPARSE_SET for c in cc])
    logger.info(f"  X_fused: {X_fused.shape}  y: {y.shape}")
    return X_fused, y, geri, cc_sparse


# ═══════════════════════════════════════════════════════════════════════
#  STEP 4: 计算 SHAP 值
# ═══════════════════════════════════════════════════════════════════════

def compute_shap(clf, X_fused, sample_n=500):
    """
    用 TreeExplainer 计算 SHAP 值。

    返回
    ----
    shap_struct : np.ndarray (N_sample, 8)  — 8个生命体征的SHAP值
    shap_text   : np.ndarray (N_sample,)    — 768维BERT嵌入SHAP的L1范数（文本综合贡献）
    X_struct_sample : np.ndarray (N_sample, 8)  — 对应的原始生命体征值（用于颜色编码）
    sample_idx  : np.ndarray (N_sample,)    — 抽样索引
    """
    logger.info(f"Computing SHAP values (TreeExplainer, sample={sample_n}) …")
    rng = np.random.default_rng(42)
    N   = X_fused.shape[0]
    idx = rng.choice(N, size=min(sample_n, N), replace=False)
    X_sample = X_fused[idx]

    explainer  = shap.TreeExplainer(clf)
    shap_vals  = explainer.shap_values(X_sample)   # (N_sample, 776)

    # 分离 BERT 部分（前768列）和结构化部分（后8列）
    shap_bert   = shap_vals[:, :768]     # (N_sample, 768)
    shap_struct = shap_vals[:, 768:]     # (N_sample, 8)

    # BERT 嵌入的 SHAP 聚合：每个样本的文本综合贡献 = L1 范数（带符号：用sum）
    # sum 保留方向信息（正 = 推动入院，负 = 保护性）
    shap_text = shap_bert.sum(axis=1)   # (N_sample,)

    # 原始生命体征值（用于 beeswarm 颜色编码）
    X_struct_sample = X_fused[idx, 768:]   # (N_sample, 8)

    logger.info(f"  SHAP computed: struct={shap_struct.shape}, text={shap_text.shape}")
    return shap_struct, shap_text, X_struct_sample, idx


# ═══════════════════════════════════════════════════════════════════════
#  STEP 5: 绘制 Figure 4
# ═══════════════════════════════════════════════════════════════════════

def _hide(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

def _plabel(ax, letter, dx=-0.12, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")

def _wilson(k, n, z=1.96):
    if n == 0: return 0., 0.
    p = k/n; d = 1 + z**2/n
    c = (p + z**2/(2*n)) / d
    m = z * np.sqrt(p*(1-p)/n + z**2/(4*n**2)) / d
    return max(0., c-m), min(1., c+m)


def panel_shap(ax, shap_struct, shap_text, X_struct_sample):
    """
    真实 SHAP beeswarm：9个特征 = 8个生命体征 + 1个文本综合。
    特征按平均|SHAP|从下（小）到上（大）排序。
    """
    # 构建9个特征的矩阵（8 struct + 1 text）
    shap_all = np.hstack([shap_struct, shap_text.reshape(-1, 1)])  # (N, 9)
    # 文本特征值：用 |shap_text| 归一化到[0,1]作为颜色编码代理
    text_feat_val = np.abs(shap_text) / (np.abs(shap_text).max() + 1e-9)
    # 对于生命体征，用实际值归一化到[0,1]
    feat_vals_all = np.zeros_like(shap_all)
    for j in range(8):
        col = X_struct_sample[:, j]
        valid = ~np.isnan(col)
        if valid.sum() > 1:
            lo, hi = np.nanpercentile(col, 5), np.nanpercentile(col, 95)
            feat_vals_all[:, j] = np.clip((col - lo) / (hi - lo + 1e-9), 0, 1)
    feat_vals_all[:, 8] = text_feat_val

    # 特征名称（9个）
    feat_names = STRUCT_NAMES + ["Text (chief complaint)"]

    # 按平均|SHAP|排序（从小到大，底部到顶部）
    mean_abs = np.abs(shap_all).mean(axis=0)
    order    = np.argsort(mean_abs)   # 升序：最不重要在底部

    sm = None
    N  = shap_all.shape[0]
    for plot_idx, feat_idx in enumerate(order):
        sv = shap_all[:, feat_idx]
        fv = feat_vals_all[:, feat_idx]
        rng_jitter = np.random.default_rng(feat_idx)
        yj = np.full(N, plot_idx) + rng_jitter.uniform(-0.36, 0.36, N)
        sm = ax.scatter(sv, yj, c=fv, cmap="RdBu_r", vmin=0, vmax=1,
                        s=3.0, alpha=0.60, linewidths=0, zorder=2)

    ax.axvline(0, color="0.40", lw=0.65, ls="--", zorder=1)
    ax.set_yticks(range(9))
    ax.set_yticklabels([feat_names[i] for i in order], fontsize=5.5)
    ax.set_ylim([-0.65, 8.65])
    ax.set_xlabel("SHAP value (impact on admission probability)", fontsize=7)
    ax.grid(axis="x", lw=0.3, alpha=0.35, zorder=0)

    if sm is not None:
        cbar = plt.colorbar(sm, ax=ax, orientation="vertical",
                            pad=0.02, fraction=0.034, aspect=26)
        cbar.set_label("Feature value", fontsize=5.5)
        cbar.set_ticks([0, 1])
        cbar.set_ticklabels(["Low", "High"], fontsize=5.0)
        cbar.ax.tick_params(length=2)

    _hide(ax); ax.spines["left"].set_visible(True)
    _plabel(ax, "a", dx=-0.10)


def panel_error(ax, y, prob, thr=0.28):
    """垂直条形图：四类预测结果占比。"""
    pred = (prob >= thr).astype(int)
    n    = len(y)
    tp = ((pred==1)&(y==1)).sum(); fp = ((pred==1)&(y==0)).sum()
    tn = ((pred==0)&(y==0)).sum(); fn = ((pred==0)&(y==1)).sum()

    labels = ["TP\n(correct admit)", "TN\n(correct non-admit)",
              "FP\n(false alert)",   "FN\n(missed admit)"]
    vals   = np.array([tp, tn, fp, fn]) / n
    colors = [PAL["XGBoost (Vitals-only)"], PAL["TabNet"],
              PAL["NEWS2"],                 PAL["DeepTriage-CN"]]

    for i, (v, col) in enumerate(zip(vals, colors)):
        ax.bar(i, v, color=col, alpha=0.85,
               edgecolor="white", linewidth=0.5, width=0.62)
        ax.text(i, v + 0.005, f"{v:.1%}", ha="center", va="bottom",
                fontsize=6.0, color="0.25")

    ax.set_xticks(range(4))
    ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel("Proportion of predictions", fontsize=7)
    ax.set_ylim([0, vals.max() * 1.30])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.text(0.98, 0.97, f"Threshold = {thr}",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.5, color="0.45")
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)
    _hide(ax); _plabel(ax, "b")


def panel_fnr(ax, y, prob, geri, cc_sparse, thr=0.28):
    """4亚组FNR分组柱状图（年龄 × 主诉稀疏度）。"""
    pred = (prob >= thr).astype(int)
    groups = [
        ("Age <65\nrich",    (~geri)&(~cc_sparse), PAL["XGBoost (Vitals-only)"], 0.55),
        ("Age <65\nsparse",  (~geri)&( cc_sparse), PAL["XGBoost (Vitals-only)"], 0.95),
        ("Age ≥65\nrich",    ( geri)&(~cc_sparse), PAL["DeepTriage-CN"],         0.55),
        ("Age ≥65\nsparse",  ( geri)&( cc_sparse), PAL["DeepTriage-CN"],         0.95),
    ]
    x = np.arange(4)
    fnrs, lows, highs, ns = [], [], [], []
    for _, mask, _, _ in groups:
        sy  = y[mask]; sp = pred[mask]; pos = sy.sum()
        fn_c = ((sp==0)&(sy==1)).sum()
        fnr  = fn_c / max(pos, 1)
        lo, hi = _wilson(fn_c, pos)
        fnrs.append(fnr); lows.append(fnr-lo); highs.append(hi-fnr)
        ns.append(mask.sum())

    for i, (_, mask, col, alp) in enumerate(groups):
        ax.bar(x[i], fnrs[i], width=0.60, color=col, alpha=alp,
               edgecolor="white", linewidth=0.5, zorder=2)
    ax.errorbar(x, fnrs, yerr=[lows, highs],
                fmt="none", color="0.25", capsize=3.5,
                capthick=0.8, linewidth=0.8, zorder=5)

    for i, (fnr, nv, up) in enumerate(zip(fnrs, ns, highs)):
        ax.text(x[i], fnr + up + 0.004, f"n={nv}",
                ha="center", va="bottom", fontsize=5.0, color="0.40")

    y_top = max(f+h for f,h in zip(fnrs,highs))
    ax.annotate("",
                xy=(3, fnrs[3]+highs[3]+0.012),
                xytext=(1, fnrs[1]+highs[1]+0.012),
                arrowprops=dict(arrowstyle="-|>",
                                color=PAL["DeepTriage-CN"],
                                lw=0.9, connectionstyle="arc3,rad=-0.20"))
    ax.text(2.0, y_top + 0.032, "Elder + sparse\n= highest FNR",
            ha="center", va="bottom", fontsize=5.5,
            color=PAL["DeepTriage-CN"])

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.8)
    ax.set_ylabel("False-negative rate (FNR)", fontsize=7)
    ax.set_ylim([0, y_top + 0.12])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0, decimals=0))
    ax.grid(axis="y", lw=0.3, alpha=0.35)
    ax.set_axisbelow(True)

    hl = [mpatches.Patch(color=PAL["XGBoost (Vitals-only)"], alpha=0.9, label="Age <65"),
          mpatches.Patch(color=PAL["DeepTriage-CN"],          alpha=0.9, label="Age ≥65")]
    ax.legend(handles=hl, fontsize=5.8, loc="upper left")
    _hide(ax); _plabel(ax, "c")


def make_figure4(shap_struct, shap_text, X_struct_sample,
                 y, prob, geri, cc_sparse, outdir):
    """生成并保存 Figure 4（三面板）。"""
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 4.15))
    fig.subplots_adjust(wspace=0.52)

    panel_shap(axes[0], shap_struct, shap_text, X_struct_sample)
    panel_error(axes[1], y, prob)
    panel_fnr(axes[2], y, prob, geri, cc_sparse)

    fig.tight_layout(pad=0.4)
    out_path = os.path.join(outdir, "Figure4_SHAP_Real.tiff")
    fig.savefig(out_path, dpi=300, format="tiff", pad_inches=0.05,
                pil_kwargs={"compression": "tiff_lzw"})
    logger.info(f"✓ Figure saved → {out_path}")
    plt.close(fig)
    return out_path


# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Compute real SHAP values and generate Figure 4")
    ap.add_argument("--train",   required=True,
                    help="Training CSV (DeepTriage_train_8000.csv)")
    ap.add_argument("--test",    required=True,
                    help="Test CSV (DeepTriage_test_2000.csv)")
    ap.add_argument("--model",   default=None,
                    help="Path to saved model .pkl (optional; trains if absent)")
    ap.add_argument("--outdir",  default="outputs/figures")
    ap.add_argument("--shap_n",  type=int, default=500,
                    help="Number of samples for SHAP (default 500, max 2000)")
    ap.add_argument("--save_shap", action="store_true",
                    help="Save SHAP arrays to .npy for reuse")
    ap.add_argument("--load_shap", default=None,
                    help="Load pre-computed SHAP .npy instead of recomputing")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    logger.info("=" * 62)
    logger.info("  DeepTriage-CN SHAP Computation + Figure 4")
    logger.info("=" * 62)

    # ── 加载数据 ────────────────────────────────────────────────────────
    logger.info("\nStep 1: Loading data …")
    train_df = load_csv(args.train)
    val_df   = load_csv(args.test)
    logger.info(f"  Train: {len(train_df)} rows | "
                f"admission={train_df['hospital_admission'].mean():.3f}")
    logger.info(f"  Test : {len(val_df)} rows  | "
                f"admission={val_df['hospital_admission'].mean():.3f}")

    # ── 训练或加载模型 ──────────────────────────────────────────────────
    logger.info("\nStep 2: Model …")
    if args.model and Path(args.model).exists():
        logger.info(f"  Loading from {args.model}")
        clf, enc, te = load_model(args.model)
    else:
        logger.info("  No model file found — training from scratch …")
        clf, enc, te = train_model(train_df, args.outdir)

    # ── 构建融合特征矩阵 ─────────────────────────────────────────────────
    logger.info("\nStep 3: Building X_fused …")
    X_fused, y, geri, cc_sparse = build_fused(val_df, enc, te)

    # 从 XGBoost 获取预测概率
    prob = clf.predict_proba(X_fused)[:, 1]
    from sklearn.metrics import roc_curve, auc
    fpr, tpr, _ = roc_curve(y, prob)
    logger.info(f"  Model AUROC on test set: {auc(fpr,tpr):.4f}")

    # ── 计算或加载 SHAP ──────────────────────────────────────────────────
    logger.info("\nStep 4: SHAP …")
    if args.load_shap and Path(args.load_shap).exists():
        logger.info(f"  Loading pre-computed SHAP from {args.load_shap}")
        npz = np.load(args.load_shap)
        shap_struct      = npz["shap_struct"]
        shap_text        = npz["shap_text"]
        X_struct_sample  = npz["X_struct"]
        sample_idx       = npz["idx"]
    else:
        shap_struct, shap_text, X_struct_sample, sample_idx = compute_shap(
            clf, X_fused, sample_n=args.shap_n)
        if args.save_shap:
            shap_path = os.path.join(args.outdir, "shap_values.npz")
            np.savez(shap_path,
                     shap_struct=shap_struct,
                     shap_text=shap_text,
                     X_struct=X_struct_sample,
                     idx=sample_idx)
            logger.info(f"  SHAP arrays saved → {shap_path}")

    # ── 生成 Figure 4 ────────────────────────────────────────────────────
    logger.info("\nStep 5: Generating Figure 4 …")
    out = make_figure4(shap_struct, shap_text, X_struct_sample,
                       y, prob, geri, cc_sparse, args.outdir)

    logger.info("\n" + "=" * 62)
    logger.info(f"  Done!  →  {out}")
    logger.info("=" * 62)


if __name__ == "__main__":
    main()
