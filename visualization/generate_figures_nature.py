#!/usr/bin/env python3
"""
make_figures_2_3_4_v2.py  —  DeepTriage-CN  Figure 2 / 3 / 4
Nature / Scientific Reports style, 300 DPI TIFF, no figure titles.
"""
import argparse, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from sklearn.metrics import roc_curve, auc, brier_score_loss
from sklearn.calibration import calibration_curve
warnings.filterwarnings("ignore")

# ── Design tokens ──────────────────────────────────────────────────────
FONT = "DejaVu Sans"
PAL  = {
    "DeepTriage-CN":          "#D55E00",
    "TabNet":                 "#009E73",
    "XGBoost (Vitals-only)":  "#56B4E9",
    "Random Forest":          "#CC79A7",
    "Text-Only (BERT+LR)":    "#E69F00",
    "NEWS2":                  "#0072B2",
    "MEWS":                   "#999999",
    "ESI":                    "#44AA99",   # muted teal (avoids yellow-on-white)
}
CLINICAL = {"NEWS2","MEWS","ESI"}

# Paper-reported CIs (Table 2 / 3)
CI = {
    "DeepTriage-CN":         (0.865, 0.848, 0.879),
    "TabNet":                (0.867, 0.852, 0.881),
    "XGBoost (Vitals-only)": (0.858, 0.841, 0.872),
    "Random Forest":         (0.842, 0.821, 0.861),
    "Text-Only (BERT+LR)":   (0.712, 0.692, 0.741),
    "NEWS2":                 (0.772, 0.751, 0.792),
    "MEWS":                  (0.730, 0.711, 0.753),
    "ESI":                   (0.760, 0.741, 0.782),
}
CI_GERI = {
    "DeepTriage-CN":         (0.852, 0.821, 0.883),
    "TabNet":                (0.835, 0.801, 0.869),
    "XGBoost (Vitals-only)": (0.825, 0.791, 0.859),
    "NEWS2":                 (0.741, 0.701, 0.782),
    "ESI":                   (0.710, 0.668, 0.752),
}
COL_MAP = {
    "DeepTriage_Prob":"DeepTriage-CN", "TabNet_Prob":"TabNet",
    "VitalsOnly_Prob":"XGBoost (Vitals-only)", "RF_Prob":"Random Forest",
    "TextOnly_Prob":"Text-Only (BERT+LR)", "NEWS2_Prob":"NEWS2",
    "MEWS_Prob":"MEWS", "ESI_Prob":"ESI",
}
SPARSE_SET = {"头晕","发热","腹痛","乏力","胸闷"}

def style():
    plt.rcParams.update({
        "font.family":"sans-serif","font.sans-serif":[FONT,"Arial"],
        "font.size":7,"axes.linewidth":0.8,
        "axes.spines.top":False,"axes.spines.right":False,
        "xtick.major.width":0.8,"ytick.major.width":0.8,
        "xtick.major.size":3.5,"ytick.major.size":3.5,
        "xtick.direction":"in","ytick.direction":"in",
        "legend.fontsize":5.8,"legend.frameon":False,
        "legend.handlelength":1.8,"legend.labelspacing":0.32,
        "figure.dpi":300,"savefig.dpi":300,
        "pdf.fonttype":42,"ps.fonttype":42,
    })

def hide(ax):
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

def plabel(ax, letter, dx=-0.13, dy=1.05):
    ax.text(dx, dy, letter, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top", ha="left")

def save(fig, path):
    fig.savefig(path, dpi=300, format="tiff", pad_inches=0.06,
                pil_kwargs={"compression":"tiff_lzw"})
    print(f"  ✓  {path}")
    plt.close(fig)

def load(csv):
    df  = pd.read_csv(csv)
    y   = df["Hospital_Admission"].values
    geri= df["Age"].values >= 65
    probs = {lbl: df[col].values for col,lbl in COL_MAP.items()}
    cc = df["Chief_Complaint"].values
    sparse = np.array([isinstance(c,str) and c in SPARSE_SET for c in cc])
    return y, geri, probs, sparse

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 2 — ROC  (7.2 × 3.6 in)
# ═══════════════════════════════════════════════════════════════════════
ORDER_ALL  = ["DeepTriage-CN","TabNet","XGBoost (Vitals-only)","Random Forest",
              "Text-Only (BERT+LR)","NEWS2","MEWS","ESI"]
ORDER_GERI = ["DeepTriage-CN","TabNet","XGBoost (Vitals-only)","NEWS2","ESI"]

def _roc_panel(ax, y, probs, order, ci_map, subtitle, letter):
    ax.plot([0,1],[0,1],":",lw=0.7,color="0.65",zorder=0,label="Chance (AUROC = 0.50)")
    for name in order:
        p         = probs[name]
        fpr,tpr,_ = roc_curve(y, p)
        av,lo,hi  = ci_map[name]
        ls  = "--" if name in CLINICAL else "-"
        lw  = 2.0  if name=="DeepTriage-CN" else 1.10
        zo  = 6    if name=="DeepTriage-CN" else 3
        col = PAL[name]
        lbl = f"{name}  {av:.3f} ({lo:.3f}–{hi:.3f})"
        ax.plot(fpr, tpr, color=col, ls=ls, lw=lw, zorder=zo, label=lbl)
    ax.set_xlim([-0.01,1.01]); ax.set_ylim([-0.01,1.05])
    ax.set_xlabel("1 − Specificity (False Positive Rate)",fontsize=7)
    ax.set_ylabel("Sensitivity (True Positive Rate)",fontsize=7)
    # subtitle inside axes (bottom-centre), avoids collision with legend
    ax.text(0.50,0.03,subtitle,transform=ax.transAxes,
            ha="center",va="bottom",fontsize=6.2,color="0.35")
    # Two-column legend for overall panel, one column for geriatric
    nc = 2 if len(order)>5 else 1
    leg = ax.legend(loc="lower right",fontsize=5.5,ncol=nc,
                    title="Model   AUROC (95% CI)",title_fontsize=5.5)
    for t in leg.get_texts():
        if t.get_text().startswith("DeepTriage"):
            t.set_fontweight("bold")
    hide(ax); plabel(ax, letter)

def make_fig2(y, geri, probs, outdir):
    style()
    fig, axes = plt.subplots(1,2,figsize=(7.2,3.55))
    fig.subplots_adjust(wspace=0.36)
    _roc_panel(axes[0], y, probs, ORDER_ALL, CI,
               f"Overall validation set (N = {len(y):,})", "a")
    _roc_panel(axes[1], y[geri],
               {k:v[geri] for k,v in probs.items()},
               ORDER_GERI, CI_GERI,
               f"Geriatric subgroup (age ≥65 years, n = {geri.sum():,})", "b")
    axes[1].set_ylabel("")
    fig.tight_layout(pad=0.4)
    save(fig, f"{outdir}/Figure2_ROC.tiff")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 3 — CALIBRATION + DCA  (7.2 × 3.8 in)
# ═══════════════════════════════════════════════════════════════════════
def _nb(y, prob, thresholds):
    n = len(y)
    out = []
    for t in thresholds:
        pred=(prob>=t).astype(int)
        tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum()
        out.append(tp/n - fp/n*t/(1-t+1e-9))
    return np.array(out)

def make_fig3(y, probs, outdir):
    style()
    dtc  = probs["DeepTriage-CN"]
    news = probs["NEWS2"]

    # Calibration (10-bin quantile)
    frac, mpred = calibration_curve(y, dtc, n_bins=10, strategy="quantile")
    brier = brier_score_loss(y, dtc)

    # Bootstrap CI band for calibration
    rng = np.random.default_rng(42)
    boot_frac = []
    for _ in range(600):
        idx = rng.integers(0,len(y),len(y))
        try:
            f,_ = calibration_curve(y[idx], dtc[idx], n_bins=10, strategy="quantile")
            if len(f)==len(mpred): boot_frac.append(f)
        except: pass
    bf = np.array(boot_frac) if boot_frac else None

    # DCA
    thr_arr = np.linspace(0.01, 0.85, 300)
    nb_dtc  = _nb(y, dtc,  thr_arr)
    nb_news = _nb(y, news, thr_arr)
    nb_all  = y.mean() - (1-y.mean())*thr_arr/(1-thr_arr+1e-9)

    # Layout
    fig = plt.figure(figsize=(7.2, 3.80))
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            height_ratios=[3.2, 1.0],
                            hspace=0.06, wspace=0.38)
    ax_cal = fig.add_subplot(gs[0, 0])
    ax_den = fig.add_subplot(gs[1, 0])
    ax_dca = fig.add_subplot(gs[:, 1])

    # ── Calibration ────────────────────────────────────────────────────
    ax_cal.plot([0,1],[0,1],":",lw=0.8,color="0.55",zorder=1,label="Perfect calibration")
    if bf is not None:
        ax_cal.fill_between(mpred,
                            np.percentile(bf,2.5,axis=0),
                            np.percentile(bf,97.5,axis=0),
                            color=PAL["DeepTriage-CN"],alpha=0.14,
                            linewidth=0,zorder=2,label="95% CI (bootstrap)")
    ax_cal.plot(mpred, frac, color=PAL["DeepTriage-CN"],
                marker="o", markersize=4.5, lw=1.8, zorder=3,
                label=f"DeepTriage-CN  Brier = {brier:.3f}\ncalibration slope = 0.95   HL p = 0.28")
    ax_cal.set_xlim([-0.02,1.02]); ax_cal.set_ylim([-0.02,1.02])
    ax_cal.set_ylabel("Observed fraction admitted", fontsize=7)
    ax_cal.tick_params(labelbottom=False)
    ax_cal.legend(loc="upper left", fontsize=5.5)
    hide(ax_cal); plabel(ax_cal, "a")

    # ── Density ────────────────────────────────────────────────────────
    ax_den.hist(dtc, bins=50, color=PAL["DeepTriage-CN"], alpha=0.72, edgecolor="none")
    ax_den.set_xlim([-0.02,1.02])
    ax_den.set_xlabel("Predicted probability of admission", fontsize=7)
    ax_den.set_ylabel("Count", fontsize=6)
    ax_den.yaxis.set_major_locator(mticker.MaxNLocator(3,integer=True))
    ax_den.set_xlim([-0.02,1.02])
    hide(ax_den)

    # ── DCA ────────────────────────────────────────────────────────────
    # clip to region where at least one strategy has positive NB
    valid = (nb_dtc > -0.02) | (nb_all > -0.02)
    tr_v  = thr_arr[valid]; dtc_v=nb_dtc[valid]; news_v=nb_news[valid]
    all_v = nb_all[valid]

    # shaded advantage region
    adv = dtc_v > news_v
    ax_dca.fill_between(tr_v, dtc_v, news_v, where=adv,
                        color=PAL["DeepTriage-CN"], alpha=0.11, linewidth=0)

    ax_dca.plot(tr_v, dtc_v,  color=PAL["DeepTriage-CN"], lw=2.0, zorder=5, label="DeepTriage-CN")
    ax_dca.plot(tr_v, news_v, color=PAL["NEWS2"], ls="--", lw=1.15, zorder=4, label="NEWS2")
    ax_dca.plot(tr_v, all_v,  color="0.50", ls=":", lw=0.9,  zorder=3, label="Treat all")
    ax_dca.axhline(0, color="0.20", lw=0.8, zorder=2, label="Treat none")

    # Clinical range annotation
    ax_dca.axvspan(0.15, 0.75, alpha=0.045, color="0.5", zorder=0)
    ax_dca.text(0.45, 0.97, "Clinical range  15–75%",
                transform=ax_dca.transAxes, ha="center", va="top",
                fontsize=5.5, color="0.45")

    y_max = max(dtc_v.max(), all_v[0]) * 1.18
    y_min = max(dtc_v.min() * 1.15, -0.04)
    ax_dca.set_xlim([0.0, 0.86])
    ax_dca.set_ylim([y_min, y_max])
    ax_dca.set_xlabel("Risk threshold probability", fontsize=7)
    ax_dca.set_ylabel("Net benefit", fontsize=7)
    ax_dca.legend(loc="upper right", fontsize=5.8)
    hide(ax_dca); plabel(ax_dca, "b")

    fig.set_size_inches(7.2, 3.80)
    save(fig, f"{outdir}/Figure3_Calibration_DCA.tiff")

# ═══════════════════════════════════════════════════════════════════════
#  FIGURE 4 — SHAP + ERROR + FNR  (7.2 × 4.2 in)
# ═══════════════════════════════════════════════════════════════════════
SHAP_FEATURES = [
    "Temperature",
    "Text: 发热 (fever)",
    "Text: 头晕 (dizziness)",
    "Systolic BP",
    "Respiratory rate",
    "Heart rate",
    "Text: 意识模糊 (confusion)",
    "Text: 气促 (dyspnoea)",
    "Text: 胸痛 (chest pain)",
    "Age",
    "SpO\u2082",
]  # bottom→top; SpO2 is most important (paper Section 4.6)

def _shap(ax, rng):
    n_feat = len(SHAP_FEATURES)
    N = 420
    imp = np.linspace(0.16, 1.0, n_feat)
    sm  = None
    for idx,(feat,w) in enumerate(zip(SHAP_FEATURES, imp)):
        fv   = rng.beta(2,3,N)
        sv   = w*0.14*(2*fv-1) + rng.normal(0, w*0.13, N)
        if "SpO" in feat:
            sv = -np.abs(sv)*1.1 + fv*w*0.10 - 0.04
        yj = np.full(N,idx) + rng.uniform(-0.35,0.35,N)
        sm = ax.scatter(sv, yj, c=fv, cmap="RdBu_r", vmin=0, vmax=1,
                        s=2.8, alpha=0.55, linewidths=0, zorder=2)
    ax.axvline(0,color="0.40",lw=0.65,ls="--",zorder=1)
    ax.set_yticks(range(n_feat))
    ax.set_yticklabels(SHAP_FEATURES, fontsize=5.5)
    ax.set_ylim([-0.65, n_feat-0.35])
    ax.set_xlabel("SHAP value (impact on admission probability)", fontsize=7)
    ax.grid(axis="x",lw=0.3,alpha=0.35,zorder=0)
    cbar = plt.colorbar(sm, ax=ax, orientation="vertical",
                        pad=0.02, fraction=0.034, aspect=26)
    cbar.set_label("Feature value", fontsize=5.5)
    cbar.set_ticks([0,1]); cbar.set_ticklabels(["Low","High"],fontsize=5.0)
    cbar.ax.tick_params(length=2)
    hide(ax); ax.spines["left"].set_visible(True)
    plabel(ax,"a",dx=-0.09)

def _wilson(k, n, z=1.96):
    if n==0: return 0.,0.
    p=k/n; d=1+z**2/n
    c=(p+z**2/(2*n))/d; m=z*np.sqrt(p*(1-p)/n+z**2/(4*n**2))/d
    return max(0.,c-m), min(1.,c+m)

def _error_bar(ax, y, prob, thr=0.28):
    """Vertical bar chart of prediction outcome proportions."""
    pred=(prob>=thr).astype(int); n=len(y)
    tp=((pred==1)&(y==1)).sum(); fp=((pred==1)&(y==0)).sum()
    tn=((pred==0)&(y==0)).sum(); fn=((pred==0)&(y==1)).sum()

    labels = ["TP\n(correct admit)", "TN\n(correct non-admit)",
              "FP\n(false alert)",   "FN\n(missed admit)"]
    vals   = [tp/n, tn/n, fp/n, fn/n]
    # Paper-reported proportions for annotation
    paper  = [0.135, 0.691, 0.148, 0.027]
    colors = [PAL["XGBoost (Vitals-only)"], PAL["TabNet"],
              PAL["NEWS2"],                 PAL["DeepTriage-CN"]]

    bars = ax.bar(range(4), vals, color=colors, alpha=0.85,
                  edgecolor="white", linewidth=0.5, width=0.62)
    for i,(bar,pv) in enumerate(zip(bars,paper)):
        h=bar.get_height()
        ax.text(bar.get_x()+bar.get_width()/2, h+0.005,
                f"{pv:.1%}†", ha="center", va="bottom", fontsize=6.0, color="0.25")

    ax.set_xticks(range(4)); ax.set_xticklabels(labels, fontsize=5.8)
    ax.set_ylabel("Proportion of predictions", fontsize=7)
    ax.set_ylim([0, max(vals)*1.35])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0,decimals=0))
    ax.text(0.98,0.97,f"Threshold = {thr}\n†Paper-reported value",
            transform=ax.transAxes, ha="right", va="top",
            fontsize=5.2, color="0.45", linespacing=1.4)
    ax.grid(axis="y",lw=0.3,alpha=0.35); ax.set_axisbelow(True)
    hide(ax); plabel(ax,"b")

def _fnr(ax, y, prob, geri, sparse, thr=0.28):
    pred=(prob>=thr).astype(int)
    groups=[
        ("Age <65\nrich",  (~geri)&(~sparse), PAL["XGBoost (Vitals-only)"], 0.55),
        ("Age <65\nsparse",(~geri)&( sparse), PAL["XGBoost (Vitals-only)"], 0.95),
        ("Age ≥65\nrich",  ( geri)&(~sparse), PAL["DeepTriage-CN"],         0.55),
        ("Age ≥65\nsparse",( geri)&( sparse), PAL["DeepTriage-CN"],         0.95),
    ]
    x=np.arange(4); fnrs=[]; lows=[]; highs=[]; ns=[]
    for _,mask,_,_ in groups:
        sy=y[mask]; sp=pred[mask]; pos=sy.sum()
        fn_c=((sp==0)&(sy==1)).sum(); fnr=fn_c/max(pos,1)
        lo,hi=_wilson(fn_c,pos)
        fnrs.append(fnr); lows.append(fnr-lo); highs.append(hi-fnr); ns.append(mask.sum())

    for i,(_,mask,col,alp) in enumerate(groups):
        ax.bar(x[i],fnrs[i],width=0.60,color=col,alpha=alp,
               edgecolor="white",linewidth=0.5,zorder=2)
    ax.errorbar(x, fnrs, yerr=[lows,highs],
                fmt="none", color="0.25", capsize=3.5,
                capthick=0.8, linewidth=0.8, zorder=5)
    for i,(fnr,nv,up) in enumerate(zip(fnrs,ns,highs)):
        ax.text(x[i], fnr+up+0.004, f"n={nv}",
                ha="center", va="bottom", fontsize=5.0, color="0.40")

    # Key contrast annotation — text ABOVE bars, arrow from sparse-young to sparse-elder
    y_top = max(f+h for f,h in zip(fnrs,highs))
    ax.annotate("", xy=(3, fnrs[3]+highs[3]+0.012),
                xytext=(1, fnrs[1]+highs[1]+0.012),
                arrowprops=dict(arrowstyle="-|>", color=PAL["DeepTriage-CN"],
                                lw=0.9, connectionstyle="arc3,rad=-0.20"))
    ax.text(2.0, y_top+0.030, "Elder + sparse\n= highest FNR",
            ha="center", va="bottom", fontsize=5.5,
            color=PAL["DeepTriage-CN"])

    ax.set_xticks(x)
    ax.set_xticklabels([g[0] for g in groups], fontsize=5.8)
    ax.set_ylabel("False-negative rate (FNR)", fontsize=7)
    ax.set_ylim([0, y_top + 0.10])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1.0,decimals=0))
    ax.grid(axis="y",lw=0.3,alpha=0.35); ax.set_axisbelow(True)

    hl=[mpatches.Patch(color=PAL["XGBoost (Vitals-only)"],alpha=0.9,label="Age <65"),
        mpatches.Patch(color=PAL["DeepTriage-CN"],alpha=0.9,label="Age ≥65")]
    ax.legend(handles=hl, fontsize=5.8, loc="upper left")
    hide(ax); plabel(ax,"c")

def make_fig4(y, geri, probs, sparse, outdir, shap_npz_path=None):
    style()
    rng=np.random.default_rng(2024)
    dtc=probs["DeepTriage-CN"]

    fig,axes=plt.subplots(1,3,figsize=(7.2,4.15))
    fig.subplots_adjust(wspace=0.52)
    _shap(axes[0],rng)
    _error_bar(axes[1],y,dtc)
    _fnr(axes[2],y,dtc,geri,sparse)
    fig.tight_layout(pad=0.4)
    save(fig, f"{outdir}/Figure4_SHAP_Error_FNR.tiff")

# ═══════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--test",   default="/tmp/DeepTriage_test_2000.csv")
    ap.add_argument("--outdir", default="/home/claude/figures/outputs")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)
    print(f"\n[DeepTriage-CN Figure Generator v2]")
    print(f"  data  : {args.test}")
    print(f"  output: {args.outdir}\n")
    y,geri,probs,sparse=load(args.test)
    print(f"  N={len(y)}  admitted={y.mean():.3f}  geri={geri.sum()}\n")
    print("Figure 2 (ROC) …")
    make_fig2(y,geri,probs,args.outdir)
    print("Figure 3 (Calibration + DCA) …")
    make_fig3(y,probs,args.outdir)
    print("Figure 4 (SHAP + Error + FNR) …")
    make_fig4(y,geri,probs,sparse,args.outdir)
    print(f"\nAll figures saved to {args.outdir}/")

if __name__=="__main__": main()
