
---

### 51. `DeepTriage-CN/docs/figure_generation_log.md`

```markdown
# Figure Generation Log

This file documents the exact scripts, dependencies, and parameters used to produce each figure in the DeepTriage-CN paper. It serves as a permanent record for computational reproducibility.

---

## General Settings

- **Config file**: `config.yaml`
- **Random seed**: `42`
- **Python**: 3.9
- **Key libraries**: `matplotlib==3.7.5`, `seaborn==0.13.0`, `shap==0.44.0`

---

## Figure 2a – Overall ROC Curves (Temporal Validation Set)

**Description**: ROC curves for DeepTriage-CN, TabNet, XGBoost (Vitals), Random Forest, Text‑Only LR, NEWS2, MEWS, ESI on the full temporal validation set (N = 2,000).

**Generating script**: `scripts/generate_all_figures.py` → calls `visualization/plot_roc.py::plot_roc_overall()`

**Required inputs**:
- Trained models in `outputs/models/`
- Validation predictions from `evaluate_all_models.py`
- Clinical scores: `outputs/models/clinical_scores_val.csv`

**Output file**: `outputs/figures/figure2a_roc_overall.tiff`

**Colour scheme** (defined in `visualization/plot_roc.py::MODEL_STYLES`):
| Model | Color | Line Style |
|---|---|---|
| DeepTriage-CN | `#E34A33` | Solid |
| TabNet | `#2C3E50` | Solid |
| XGBoost (Vitals) | `#3498DB` | Solid |
| Random Forest | `#95A5A6` | Solid |
| Text-Only | `#F39C12` | Solid |
| NEWS2 | `#8E44AD` | Dashed |
| MEWS | `#16A085` | Dashed |
| ESI | `#D35400` | Dashed |

---

## Figure 2b – Geriatric Subgroup ROC (Age ≥ 65)

**Description**: Same as Figure 2a, but restricted to patients aged ≥ 65 (n ≈ 410).

**Script**: `plot_roc_geriatric()` in `visualization/plot_roc.py`

**Output**: `outputs/figures/figure2b_roc_geriatric.tiff`

---

## Figure 3 – Calibration Plot

**Description**: Calibration curve of DeepTriage‑CN with density histogram of predicted probabilities.

**Script**: `plot_calibration_curve()` in `visualization/plot_calibration.py`

**Output**: `outputs/figures/figure3_calibration.tiff`

**Brier score**: computed by `evaluation/calibration.py` and displayed on the plot.

---

## Figure 4 – Decision Curve Analysis

**Description**: Net benefit curves for DeepTriage‑CN and NEWS2 versus “Treat All” and “Treat None” strategies.

**Script**: `plot_decision_curve()` in `visualization/plot_decision_curve.py`

**Output**: `outputs/figures/figure4_decision_curve.tiff`

**Threshold range**: 1% – 99% probability.

---

## Figure 5 – SHAP Beeswarm Summary Plot

**Description**: Top 20 feature attributions for the XGBoost classifier of DeepTriage‑CN.

**Script**: `plot_shap_beeswarm()` in `visualization/plot_shap.py`

**Input**:
- Trained XGBoost model from `deeptriage_cn.pkl`
- Fused feature matrix (text embeddings + standardized structured features)

**Output**: `outputs/figures/figure5_shap.tiff`

**Notes**: SHAP values are purely associative and not interpreted as causal evidence.

---

## Figure 6a – Error Distribution

**Description**: Bar chart of true positives, true negatives, false positives, and false negatives at the Youden‑optimal threshold (0.28).

**Script**: `plot_error_distribution()` in `visualization/plot_error_analysis.py`

**Output**: `outputs/figures/figure6a_error_distribution.tiff`

---

## Figure 6b – False‑Negative Rate by Age and Narrative Length

**Description**: Bar chart showing false‑negative rates stratified by age group (<65 vs ≥65) and narrative sparseness (≤3 vs >3 words). Includes 95% confidence intervals.

**Script**: `plot_fnr_by_subgroup()` in `visualization/plot_error_analysis.py`

**Output**: `outputs/figures/figure6b_fnr_subgroup.tiff`

---

## Supplementary Figure 1 – Robustness Under Data Degradation

**Description**: Grouped bar chart comparing baseline vs degraded (30% MNAR + noise) AUROC for DeepTriage‑CN, TabNet, and Vitals‑only XGBoost.

**Script**: `plot_robustness_bars()` in `visualization/plot_robustness.py`

**Output**: `outputs/figures/supp_figure1_robustness.tiff`

**Degradation parameters**:  
- Missing proportion: 30%  
- Noise σ multiplier: 0.5  
- MNAR logic: `src/robustness.py::mnar_missingness`

---

## Figure Naming Convention

| Figure in Paper | File Name |
|---|---|
| Figure 2a | `figure2a_roc_overall.tiff` |
| Figure 2b | `figure2b_roc_geriatric.tiff` |
| Figure 3 | `figure3_calibration.tiff` |
| Figure 4 | `figure4_decision_curve.tiff` |
| Figure 5 | `figure5_shap.tiff` |
| Figure 6a | `figure6a_error_distribution.tiff` |
| Figure 6b | `figure6b_fnr_subgroup.tiff` |
| Supplementary Figure 1 | `supp_figure1_robustness.tiff` |

All figures are rendered at **300 dpi** in **TIFF** format for publication quality.

---

*Log generated: 2025-05-07*  
*Maintainer: Wenjia Lin & Xiaolei Huang*