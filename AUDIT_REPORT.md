# DeepTriage-CN 代码审查诊断报告

> 审查人：Claude Sonnet 4.6 | 日期：2025-05-18

---

## 一、严重 Bug（会导致运行崩溃）

| # | 文件 | 问题 | 严重程度 |
|---|------|------|----------|
| B1 | `src/clinical_scores.py` | `compute_esi_level` 中使用了海象运算符 `:=` 进行赋值（`respiratory_rate := rr`），这是一个语法错误，且逻辑运算符优先级错误导致 `rr < 8` 与 `rr > 36` 的布尔表达式被错误绑定 | 🔴 严重 |
| B2 | `visualization/plot_decision_curve.py` | `from ..evaluation.decision_curve import compute_net_benefit` 使用相对导入 `..`，但 `visualization` 与 `evaluation` 是同级包（非父子关系），运行时抛出 `ImportError` | 🔴 严重 |
| B3 | `scripts/evaluate_all_models.py` | `deeptriage = joblib.load(...)` 直接反序列化，但 `fusion_model.py` 的 `save()` 存储的是一个**字典**（含 xgb_params、classifier、structured_encoder），而非 `DeepTriageCN` 对象，导致后续调用 `.predict_proba()` 抛出 `AttributeError` | 🔴 严重 |
| B4 | `scripts/generate_all_figures.py` 同 B3 | `scripts/robustness_simulation.py` 同 B3 | 🔴 严重 |
| B5 | `scripts/generate_all_figures.py` | `fnr_data` 传入的值为 `float`，但 `plot_fnr_by_subgroup` 对其做 `fnr, n = fnr_data.get(key)` 解包，抛出 `TypeError: cannot unpack non-iterable float` | 🔴 严重 |
| B6 | 全仓库文件名 | 所有 Python 脚本文件名含空格（如 `train all models.py`），但 `run_full_pipeline.py` 调用时使用下划线（`train_all_models.py`），导致管道一键运行完全失败 | 🔴 严重 |

---

## 二、逻辑错误（导致结果不可信）

| # | 文件 | 问题 | 严重程度 |
|---|------|------|----------|
| L1 | `scripts/train_all_models.py` | `preprocess_structured()` 返回的 `X_train_raw` **已经过 StandardScaler 标准化**，命名为 `raw` 有误导性。该数据随后被传入 `DeepTriageCN.fit()`，内部又调用 `structured_encoder.fit_transform()` 进行**二次标准化**，导致特征分布严重失真 | 🟠 高 |
| L2 | `scripts/train_all_models.py` | `TabNetWrapper.fit()` 接收的 `X_train_raw`（已标准化）再次被传入，TabNet 内部不会重复标准化，但与原始设计意图不符 | 🟠 高 |
| L3 | `scripts/robustness_simulation.py` | `scaler.transform(X_val_raw)` 对已标准化数据再次变换（与 L1 同根因）；同时鲁棒性实验仅测试 30% 缺失比例，未生成论文所述 10%/20%/30% 完整对比表格 | 🟠 高 |
| L4 | `scripts/generate_all_figures.py` | `_scaled(X_val_raw)` 函数对**验证集**重新 `fit_transform()`，造成数据泄漏（用验证集参数标准化验证集），且独立于训练时使用的 Scaler，导致 SHAP 计算的输入特征值与模型训练时的分布不一致 | 🟠 高 |
| L5 | `scripts/generate_all_figures.py` | `comp_lens = val_df["chief_complaint"].str.len()` 计算的是**字符数**，而论文（Section 4.7）定义"sparse"为词数 ≤3，应使用 `.str.split().str.len()` | 🟡 中 |
| L6 | `src/structured_encoder.py` | 注释称"StandardScaler 可以忽略 NaN"，实为错误说明。sklearn 默认 StandardScaler **不处理 NaN**，直接调用会抛出 `ValueError`，须改用 NaN 安全的实现 | 🟠 高 |

---

## 三、文档与规范问题

| # | 文件 | 问题 |
|---|------|------|
| D1 | `data/feature_names.json` | 文件内容被 Markdown 格式的注释污染（含 `---` 分隔符和 \`\`\`json 代码块标记），不是合法 JSON，`json.load()` 直接报错 |
| D2 | `CITATION.cff` | `cff-version: 1.1.0` 已过时，当前规范为 `1.2.0`；缺少 `doi`、`url` 等必填字段 |
| D3 | 根目录 | 缺少 `LICENSE` 文件（README 中承诺 MIT License） |
| D4 | 根目录 | README 中 `inference_single_patient.py` 与实际脚本名 `predict_single_patient.py` 不一致 |
| D5 | `docs/reproducibility_guide.md` | 引用 `environment.yml` 但实际文件名为 `environment.yaml` |
| D6 | 全仓库 | 缺少 `outputs/` 目录占位文件 `.gitkeep`，克隆后目录结构不完整 |

---

## 四、修复摘要（共修改/新增文件）

以下文件已被完整重写/修复，可直接覆盖上传至 GitHub：

| 文件 | 操作 |
|------|------|
| `src/preprocess.py` | 修复变量命名，区分 raw/standardized |
| `src/structured_encoder.py` | 修复 NaN 安全标准化 |
| `src/fusion_model.py` | 修复 save/load 逻辑（存储完整对象而非字典），修复双重标准化 |
| `src/clinical_scores.py` | 修复 `compute_esi_level` 中的海象运算符语法错误 |
| `scripts/train_all_models.py` | 修复数据流（传递 raw 数据而非预标准化数据） |
| `scripts/evaluate_all_models.py` | 修复模型加载逻辑 |
| `scripts/robustness_simulation.py` | 修复模型加载、双重标准化、完整 10%/20%/30% 测试 |
| `scripts/generate_all_figures.py` | 修复 `_scaled` 数据泄漏、`fnr_data` 格式、词数计算 |
| `scripts/run_full_pipeline.py` | 修复脚本文件名引用 |
| `visualization/plot_decision_curve.py` | 修复相对导入错误 |
| `data/feature_names.json` | 清除 Markdown 污染，还原为合法 JSON |
| `CITATION.cff` | 升级至 CFF 1.2.0 |
| `LICENSE` | 新增 MIT License 全文 |
| `README.md` | 全量重写，符合高水平 SCI 开源规范 |
