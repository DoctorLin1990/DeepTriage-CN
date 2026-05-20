# DeepTriage-CN

**Integrating Clinical Text with Vital Signs for Emergency Department Admission Prediction in an Aging Population**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20073330.svg)](https://doi.org/10.5281/zenodo.20073330)
[![Code Style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![TRIPOD+AI](https://img.shields.io/badge/Reporting-TRIPOD%2BAI%202024-brightgreen)](https://doi.org/10.1136/bmj-2023-078378)

> **Paper**: Lin W, Chen W, Wei G, Huang X. *DeepTriage-CN: Integrating Clinical Text with Vital Signs for Emergency Department Admission Prediction in an Aging Population.* **Scientific Reports**, 2025.
>
> **Corresponding author**: Xiaolei Huang — MichaelHuangDoctor@163.com
>
> **Ethics**: Approved by the IRB of The Second Affiliated Hospital of Fujian Medical University (No. FMU-AH-2023-699). Conducted in accordance with TRIPOD+AI 2024 guidelines.

---

## Table of Contents

1. [Overview](#1-overview)
2. [Key Results](#2-key-results)
3. [Model Architecture](#3-model-architecture)
4. [Requirements](#4-requirements)
5. [Project Structure](#5-project-structure)
6. [Data Availability Statement](#6-data-availability-statement)
7. [Quickstart — One-Command Reproduction](#7-quickstart--one-command-reproduction)
8. [Step-by-Step Usage](#8-step-by-step-usage)
9. [Configuration](#9-configuration)
10. [Output Description](#10-output-description)
11. [Code–Paper Cross-Reference](#11-codepaper-cross-reference)
12. [Bug Fixes Applied](#12-bug-fixes-applied)
13. [Ethical Statement and Limitations](#13-ethical-statement-and-limitations)
14. [Citation](#14-citation)
15. [Contact](#15-contact)

---

## 1. Overview

DeepTriage-CN is a **multimodal late-fusion framework** for predicting hospital admission from emergency department triage data. It combines:

- **Text tower** — a frozen `bert-base-chinese` encoder that extracts a 768-dimensional `[CLS]` embedding from nurse-recorded chief complaints in Mandarin Chinese.
- **Structured tower** — eight triage vital signs and demographics (age, sex, temperature, heart rate, respiratory rate, SBP, DBP, SpO₂), NaN-safely standardised with training-set parameters.
- **Fusion** — late concatenation into a 776-dimensional vector, classified by an XGBoost model.

The model was trained on **8,000 adult ED visits** from a tertiary academic medical centre in Southeast China (2023) and evaluated on a **temporal validation set of 2,000 encounters** (January–March 2024), with particular attention to the **geriatric subgroup** (age ≥ 65 years), where conventional clinical scores (NEWS2, ESI) underperform due to atypical presentations.

> **Interpretability caveat**: SHAP values included in this repository describe local mathematical associations within the learned model and do **not** confer physiological causality or clinical actionability (Section 2.4 of paper).

---

## 2. Key Results

| Model | AUROC (95% CI) | AUPRC | Sensitivity* | Specificity* |
|-------|---------------|-------|-------------|-------------|
| **DeepTriage-CN** | **0.865 (0.848–0.879)** | 0.810 | 0.84 | 0.76 |
| TabNet (Structured DL) | 0.867 (0.852–0.881) | 0.800 | 0.82 | 0.75 |
| XGBoost (Vitals-only) | 0.858 (0.841–0.872) | 0.750 | 0.78 | 0.73 |
| Random Forest | 0.842 (0.821–0.861) | 0.730 | 0.76 | 0.72 |
| Text-Only (BERT+LR) | 0.712 (0.692–0.741) | 0.540 | 0.61 | 0.68 |
| NEWS2 | 0.772 (0.751–0.792) | 0.640 | 0.69 | 0.73 |
| ESI | 0.760 (0.741–0.782) | 0.680 | 0.65 | 0.77 |

*At independently optimised Youden-optimal threshold (0.28 for DeepTriage-CN). 95% CIs from 1,000 bootstrap resamples.

**Robustness (30% MNAR + Gaussian noise, σ=0.5):** DeepTriage-CN retained **95.4%** of baseline AUROC vs. 83.2% (TabNet) and 82.8% (XGBoost).

**Geriatric subgroup (age ≥65, n=410):** DeepTriage-CN AUROC = 0.852 vs. ESI = 0.710 and NEWS2 = 0.741.

**Calibration:** Brier score = 0.12, slope = 0.95, intercept = 0.02, Hosmer–Lemeshow p = 0.28.

**NRI vs NEWS2:** 0.41 (95% CI 0.34–0.48); **NRI vs ESI:** 0.53 (95% CI 0.46–0.60).

> **Interpretation**: DeepTriage-CN's AUROC is statistically indistinguishable from TabNet (DeLong p = 0.42). The multimodal advantage is situational — greater robustness under data degradation and improved performance over vitals-only baselines in older adults — rather than a universal metric improvement.

---

## 3. Model Architecture

```
 Chief Complaint (Chinese text)          Structured Vitals & Demographics
         │                              (age, sex, temp, HR, RR, SBP, DBP, SpO₂)
         ▼                                           │
  BertTokenizer                                      ▼
  (max_length=64)                         NaN-safe z-score normalisation
         │                               (training-set parameters only)
         ▼                                           │
  Frozen BERT-Chinese                           8-d vector
  (bert-base-chinese, 110M)                         │
  torch.no_grad()                                    │
         │                                           │
  768-d [CLS] embedding ──────── concat ─────────────┘
                                     │
                              776-d fused vector
                                     │
                           XGBoost Classifier
                   (n_estimators=200, lr=0.05, max_depth=6,
                    subsample=0.8, colsample_bytree=0.8,
                    scale_pos_weight=auto)
                                     │
                      P(hospital admission) ∈ [0, 1]
```

**Text tower**: The BERT encoder is always frozen (`torch.no_grad()`). Fine-tuning the full 110M-parameter model on 8,000 samples caused severe overfitting within 3 epochs; the frozen strategy provides stable convergence (Section 3.4).

**Missing values**: Structured NaN values are preserved through the NaN-safe `StructuredEncoder` and handled natively by XGBoost's sparsity-aware split-finding. Chief complaints missing for 2.4% of visits are replaced with `[MISSING]` before BERT tokenisation.

---

## 4. Requirements

### Hardware
| | Minimum | Recommended |
|--|---------|-------------|
| RAM | 8 GB | 16 GB |
| GPU | None (CPU-only) | NVIDIA ≥ 6 GB VRAM (CUDA 11.8+) |
| Disk | 5 GB (BERT weights) | 10 GB |

### Software

**Option A — Conda (recommended):**
```bash
conda env create -f environment.yaml
conda activate deeptriage
```

**Option B — pip:**
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

**Core dependencies** (pinned in `requirements.txt`):

| Package | Version | Role |
|---------|---------|------|
| `torch` | ≥ 2.1 | BERT inference |
| `transformers` | ≥ 4.38 | `bert-base-chinese` loading |
| `xgboost` | ≥ 2.0 | Main classifier |
| `pytorch-tabnet` | ≥ 4.1 | TabNet baseline |
| `scikit-learn` | ≥ 1.4 | RF baseline, preprocessing |
| `shap` | ≥ 0.44 | SHAP explainability (Figure 5) |
| `pandas` | ≥ 2.1 | Data handling |
| `numpy` | ≥ 1.26 | Numerical computation |
| `scipy` | ≥ 1.12 | DeLong test, Hosmer–Lemeshow |
| `matplotlib` | ≥ 3.8 | Figures 2–6, Supp. Fig. 1 |
| `PyYAML` | ≥ 6.0 | Configuration parsing |

---

## 5. Project Structure

```
DeepTriage-CN/
├── config.yaml                   # All hyperparameters & paths (single source of truth)
├── requirements.txt              # Pinned Python dependencies
├── environment.yaml              # Conda environment definition
├── LICENSE                       # MIT License
├── CITATION.cff                  # CFF 1.2.0 citation metadata
├── AUDIT_REPORT.md               # Full code audit: all bugs found and fixed
│
├── src/                          # Core model source code
│   ├── __init__.py
│   ├── preprocess.py             # Data loading, cohort selection, temporal split
│   ├── structured_encoder.py     # NaN-safe z-score normaliser (StructuredEncoder)
│   ├── text_encoder.py           # Frozen BERT-Chinese [CLS] extractor (TextEncoder)
│   ├── fusion_model.py           # DeepTriageCN: two-tower late-fusion model
│   ├── tabnet_model.py           # TabNet wrapper (Section 3.6 baseline)
│   ├── baseline_models.py        # VitalsOnlyXGBoost, RandomForest, TextOnlyLR
│   ├── clinical_scores.py        # ESI, NEWS2, MEWS computation (Section 3.6)
│   └── robustness.py             # MNAR + Gaussian noise simulation (Section 3.7)
│
├── evaluation/                   # Metrics and statistical tests
│   ├── __init__.py
│   ├── metrics.py                # AUROC, AUPRC, sensitivity, specificity, F1
│   ├── bootstrap.py              # 1,000-iteration bootstrap 95% CIs (Section 3.8)
│   ├── statistical_tests.py      # DeLong test, continuous NRI (Section 3.8)
│   ├── calibration.py            # Brier score, slope/intercept, Hosmer–Lemeshow
│   ├── decision_curve.py         # Net benefit for DCA (Figure 4)
│   └── error_analysis.py         # FNR by age × narrative-length subgroup (Figure 6b)
│
├── visualization/                # Figure generation scripts
│   ├── __init__.py
│   ├── plot_roc.py               # Figure 2a (overall) and 2b (geriatric) ROC
│   ├── plot_calibration.py       # Figure 3: calibration + density sub-panel
│   ├── plot_decision_curve.py    # Figure 4: decision curve analysis
│   ├── plot_shap.py              # Figure 5: SHAP beeswarm
│   ├── plot_error_analysis.py    # Figure 6a (error distribution) and 6b (FNR)
│   └── plot_robustness.py        # Supplementary Figure 1: robustness bars
│
├── scripts/                      # End-to-end pipeline entry points
│   ├── run_full_pipeline.py      # ★ One-command reproducibility entry point
│   ├── train_all_models.py       # Train all 5 models + compute clinical scores
│   ├── evaluate_all_models.py    # Table 2 metrics, bootstrap CIs, DeLong, NRI
│   ├── robustness_simulation.py  # Data-degradation simulation (Section 3.7)
│   └── generate_all_figures.py   # All paper figures (300 DPI TIFF)
│
├── data/
│   ├── feature_names.json        # Feature schema with descriptions and missing rates
│   └── generate_synthetic_data.py# Synthetic dataset generator (smoke tests / CI)
│
├── docs/
│   └── reproducibility_guide.md  # Detailed step-by-step reproduction guide
│
├── tests/
│   └── test_smoke.py             # Smoke tests (no GPU required; pytest)
│
└── outputs/                      # All generated outputs (git-ignored except .gitkeep)
    ├── models/                   # Serialised model files
    ├── results/                  # CSV tables (Table 2, CIs, DeLong, NRI, calibration)
    ├── figures/                  # 300-dpi TIFF figures
    └── logs/                     # Training and evaluation logs
```

---

## 6. Data Availability Statement

The de-identified electronic health record dataset used in this study is **not publicly available** due to institutional privacy regulations protecting patient health information. The dataset may be requested from the corresponding author (Xiaolei Huang, MichaelHuangDoctor@163.com) subject to execution of a Data Use Agreement and subsequent Institutional Review Board approval.

To enable full computational reproducibility without patient data, this repository provides:

1. **`data/generate_synthetic_data.py`** — generates a 10,000-encounter synthetic dataset that replicates the aggregate statistical properties of the real cohort (Table 1 missingness rates, admission prevalence, age distribution). Results from synthetic data will differ from the paper's reported values.

2. **`data/feature_names.json`** — the complete data schema with feature descriptions, units, and per-variable missing rates from Table 1.

3. **Frozen BERT weights** — downloaded automatically from HuggingFace Hub on first run (`bert-base-chinese`). No manual download required.

To use your own institutional dataset, update `config.yaml → data.raw_data_path` and ensure the CSV contains the columns listed in `data/feature_names.json`.

---

## 7. Quickstart — One-Command Reproduction

```bash
# 1. Clone and enter the repository
git clone https://github.com/DoctorLin1990/DeepTriage-CN.git
cd DeepTriage-CN

# 2. Create the Conda environment
conda env create -f environment.yaml
conda activate deeptriage

# 3. Run the full pipeline on synthetic data (~20 min on CPU, ~5 min on GPU)
python scripts/run_full_pipeline.py --config config.yaml
```

This single command executes five steps in sequence:

| Step | Script | Description |
|------|--------|-------------|
| 1 | `data/generate_synthetic_data.py` | Generate 10,000-encounter synthetic dataset |
| 2 | `scripts/train_all_models.py` | Train DeepTriage-CN, TabNet, XGBoost, RF, Text-LR |
| 3 | `scripts/evaluate_all_models.py` | Compute Table 2 metrics, bootstrap CIs, DeLong, NRI |
| 4 | `scripts/robustness_simulation.py` | MNAR + noise at 10%, 20%, 30% (Supp. Fig. 1) |
| 5 | `scripts/generate_all_figures.py` | Produce all paper figures as 300 DPI TIFF |

Results are written to `outputs/results/` and figures to `outputs/figures/`.

**To skip figure generation** (faster CI):
```bash
python scripts/run_full_pipeline.py --config config.yaml --skip-figures
```

**To skip synthetic data generation** (CSV already exists):
```bash
python scripts/run_full_pipeline.py --config config.yaml --skip-datagen
```

---

## 8. Step-by-Step Usage

### 8.1 Generate (or prepare) data

```bash
# Option A: Generate synthetic dataset (no patient data required)
python data/generate_synthetic_data.py
# → writes data/synthetic_ed_visits.csv

# Option B: Use real data — ensure the CSV has these columns:
#   visit_date, patient_id, age, sex, temperature, heart_rate,
#   respiratory_rate, sbp, dbp, spo2, chief_complaint, hospital_admission
# Then update config.yaml: data.raw_data_path → your CSV path
```

### 8.2 Train all models

```bash
python scripts/train_all_models.py --config config.yaml
```

Saves to `outputs/models/`:
- `deeptriage_cn.pkl` — full DeepTriage-CN state (XGBoost + StructuredEncoder params + TextEncoder config)
- `tabnet/` — TabNet model directory
- `vitals_only_xgb.pkl`, `random_forest.pkl`, `text_only_lr.pkl` — baseline models
- `clinical_scores_val.csv` — pre-computed NEWS2, MEWS, ESI scores on the validation set

### 8.3 Evaluate all models (Table 2)

```bash
python scripts/evaluate_all_models.py --config config.yaml
```

Outputs to `outputs/results/`:
| File | Content |
|------|---------|
| `table2_metrics.csv` | AUROC, AUPRC, sensitivity, specificity, accuracy, F1 for all models |
| `bootstrap_cis.csv` | 95% bootstrap CIs (1,000 iterations) |
| `delong_tests.csv` | DeLong z and p values — DeepTriage-CN vs each comparator |
| `nri_results.csv` | Continuous NRI vs NEWS2 and ESI |
| `calibration_metrics.csv` | Brier score, slope, intercept, Hosmer–Lemeshow |

### 8.4 Robustness simulation

```bash
python scripts/robustness_simulation.py --config config.yaml
```

Tests all three MNAR proportions (10%, 20%, 30%) with Gaussian noise (σ=0.5).
Outputs: `outputs/results/robustness_results.csv` and `robustness_summary.txt`.

### 8.5 Generate all figures

```bash
python scripts/generate_all_figures.py --config config.yaml
```

Saves 300-dpi TIFF files to `outputs/figures/`:

| File | Figure | Description |
|------|--------|-------------|
| `figure2a_roc_overall.tiff` | Fig. 2a | Overall ROC curves (N=2,000) |
| `figure2b_roc_geriatric.tiff` | Fig. 2b | Geriatric subgroup ROC (n=410) |
| `figure3_calibration.tiff` | Fig. 3 | Calibration plot + density sub-panel |
| `figure4_decision_curve.tiff` | Fig. 4 | Decision Curve Analysis |
| `figure5_shap.tiff` | Fig. 5 | SHAP beeswarm feature attribution |
| `figure6a_error_distribution.tiff` | Fig. 6a | Misclassification distribution |
| `figure6b_fnr_subgroup.tiff` | Fig. 6b | FNR by age × narrative length |
| `supp_figure1_robustness.tiff` | Supp. Fig. 1 | Robustness bars (30% MNAR) |

### 8.6 Run smoke tests

```bash
pytest tests/test_smoke.py -v
```

All tests run without a GPU and complete in approximately 60 seconds.

---

## 9. Configuration

All hyperparameters are centralised in **`config.yaml`**. Key settings:

```yaml
# Data paths and temporal split
data:
  raw_data_path:         "data/synthetic_ed_visits.csv"
  training_cutoff_date:  "2024-01-01"   # Section 3.2
  validation_start_date: "2024-01-01"

# BERT encoder (Section 3.5)
text_processing:
  model_name: "bert-base-chinese"
  max_length: 64

# XGBoost (Section 3.5)
xgboost:
  n_estimators: 200    learning_rate: 0.05
  max_depth: 6         subsample: 0.8    colsample_bytree: 0.8

# TabNet (Section 3.6)
tabnet:
  n_d: 32    n_a: 32    n_steps: 5    gamma: 1.5    lambda_sparse: 0.0001

# Robustness simulation (Section 3.7)
robustness:
  missing_proportions: [0.10, 0.20, 0.30]
  noise_sigma_multiplier: 0.5

# Evaluation (Section 3.8)
evaluation:
  optimal_threshold: 0.28   # Youden-optimal from training set
  n_bootstrap: 1000
```

---

## 10. Output Description

### outputs/results/

| File | Paper section | Key values |
|------|--------------|------------|
| `table2_metrics.csv` | Table 2 | AUROC DeepTriage-CN: 0.865 |
| `bootstrap_cis.csv` | Table 2 | 95% CI for all metrics |
| `delong_tests.csv` | Section 4.2 | DeepTriage-CN vs TabNet: p=0.42 |
| `nri_results.csv` | Section 4.2 | NRI vs NEWS2: 0.41; vs ESI: 0.53 |
| `calibration_metrics.csv` | Section 4.3 | Brier=0.12, slope=0.95 |
| `robustness_results.csv` | Section 4.5 | Retention at 30%: 95.4% |

### outputs/figures/

All figures are saved as **300 DPI TIFF**, publication-ready for Scientific Reports.

---

## 11. Code–Paper Cross-Reference

| Paper section | Code location |
|---------------|--------------|
| § 3.2 Cohort selection, temporal split | `src/preprocess.py → load_and_split_data()` |
| § 3.4 Z-score normalisation, NaN handling | `src/structured_encoder.py → StructuredEncoder` |
| § 3.4 Chief complaint preprocessing, [MISSING] token | `src/preprocess.py → preprocess_text()` |
| § 3.5 Two-tower late-fusion architecture | `src/fusion_model.py → DeepTriageCN` |
| § 3.5 XGBoost hyperparameters | `config.yaml → xgboost` / `src/fusion_model.py` |
| § 3.6 TabNet hyperparameters | `config.yaml → tabnet` / `src/tabnet_model.py` |
| § 3.6 ESI, NEWS2, MEWS | `src/clinical_scores.py` |
| § 3.7 MNAR robustness simulation | `src/robustness.py → simulate_degraded_data()` |
| § 3.8 AUROC, AUPRC, sensitivity, specificity | `evaluation/metrics.py` |
| § 3.8 Bootstrap 95% CIs (n=1,000) | `evaluation/bootstrap.py` |
| § 3.8 DeLong test | `evaluation/statistical_tests.py → delong_test()` |
| § 3.8 Continuous NRI | `evaluation/statistical_tests.py → continuous_nri()` |
| § 3.8 Calibration (Brier, slope, HL) | `evaluation/calibration.py` |
| § 3.8 Decision Curve Analysis | `evaluation/decision_curve.py` |
| § 4.4 Geriatric subgroup AUROC | `scripts/generate_all_figures.py` (elder_mask) |
| § 4.5 Robustness (all 3 levels) | `scripts/robustness_simulation.py` |
| § 4.6 SHAP feature attribution | `visualization/plot_shap.py` |
| § 4.7 FNR by narrative length & age | `evaluation/error_analysis.py` |
| Figure 1 (architecture) | `src/fusion_model.py` (documented architecture) |
| Figure 2a/b (ROC curves) | `visualization/plot_roc.py` |
| Figure 3 (calibration) | `visualization/plot_calibration.py` |
| Figure 4 (DCA) | `visualization/plot_decision_curve.py` |
| Figure 5 (SHAP) | `visualization/plot_shap.py` |
| Figure 6a/b (error analysis) | `visualization/plot_error_analysis.py` |
| Supplementary Figure 1 (robustness) | `visualization/plot_robustness.py` |

---

## 12. Bug Fixes Applied

This repository corrects all issues identified in the code audit (`AUDIT_REPORT.md`):

### Critical bugs (would cause runtime crashes)

| ID | File | Fix |
|----|------|-----|
| **B1** | `src/clinical_scores.py` | Removed walrus operator `:=` inside bitwise expression in `compute_esi_level()`; replaced with correctly scoped boolean conditions. |
| **B2** | `visualization/plot_decision_curve.py` | Changed `from ..evaluation.decision_curve import compute_net_benefit` (relative import across sibling packages → `ImportError`) to an absolute import. |
| **B3/B4** | `src/fusion_model.py`, `scripts/evaluate_all_models.py`, `scripts/generate_all_figures.py`, `scripts/robustness_simulation.py` | `save()` previously stored a plain `dict`; `joblib.load()` then returned that dict, causing `AttributeError` on `.predict_proba()`. Fixed: `save()` serialises full model state; `DeepTriageCN.load_from_file()` factory method reconstructs a ready instance. |
| **B5** | `scripts/generate_all_figures.py` | `fnr_data` values were plain `float`; `plot_fnr_by_subgroup()` expected `(fnr, n)` tuples for Wilson CI computation → `TypeError`. Fixed: values are now correctly constructed as tuples. |
| **B6** | `scripts/run_full_pipeline.py` | All script filenames now use underscores (were spaces in original, breaking subprocess calls). |

### Logic errors (silently corrupt results)

| ID | File | Fix |
|----|------|-----|
| **L1** | `scripts/train_all_models.py` | `DeepTriageCN.fit()` received already-standardised arrays and then called `StructuredEncoder.fit_transform()` internally → double standardisation. Fixed: raw arrays passed via `get_raw_structured()`. |
| **L2** | `scripts/train_all_models.py` | Same double-standardisation for `TabNetWrapper.fit()`. Fixed identically. |
| **L3** | `scripts/robustness_simulation.py` | Only the 30% missingness level was tested; paper requires 10%/20%/30%. Fixed: all three proportions now iterated explicitly. |
| **L4** | `scripts/generate_all_figures.py` | `_scaled(X_val_raw)` created a fresh `StructuredEncoder` and called `fit_transform()` on validation data → data leakage. Fixed: `deeptriage.structured_encoder.transform(X_val_raw)` used instead. |
| **L5** | `scripts/generate_all_figures.py` | `comp_lens` computed character count (`.str.len()`); paper defines "sparse" as ≤ 3 *words* (Section 4.7). Fixed: `.str.split().str.len()` for word count. |
| **L6** | `src/structured_encoder.py` | Comment incorrectly stated sklearn `StandardScaler` handles NaN. Fixed: custom NaN-safe implementation using `np.nanmean` / `np.nanstd`. |

---

## 13. Ethical Statement and Limitations

- This study was approved by the IRB of The Second Affiliated Hospital of Fujian Medical University (No. FMU-AH-2023-699). Informed consent was waived for de-identified retrospective data. All procedures comply with the Declaration of Helsinki.
- **Data**: The real dataset cannot be publicly shared. The synthetic dataset in this repository replicates aggregate statistics but contains no patient information.
- **Outcome**: Hospital admission is an operational endpoint confounded by bed availability, physician risk tolerance, and systemic factors; the model learns local practice patterns rather than an objective measure of clinical severity.
- **Generalisability**: Single-centre, retrospective. Validation set limited to one winter quarter (potential seasonal case-mix bias). Not validated in non-Mandarin settings or in populations without nurse-mediated dialect translation.
- **SHAP**: All SHAP values in this repository are non-interventional mathematical associations. They do not imply causality and should not be used for bedside clinical inference.
- **Threshold**: The Youden-optimal threshold (0.28) yields PPV ≈ 0.50 at a 22% admission prevalence, implying a 1:1 true-to-false-alarm ratio. Prospective deployment requires careful threshold calibration and human-computer interaction design to mitigate alert fatigue.

---

## 14. Citation

If you use this code or findings in your work, please cite:

```bibtex
@article{lin2025deeptriage,
  title   = {{DeepTriage-CN}: Integrating Clinical Text with Vital Signs for
             Emergency Department Admission Prediction in an Aging Population},
  author  = {Lin, Wenjia and Chen, Wenliang and Wei, Guan and Huang, Xiaolei},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.5281/zenodo.20073330}
}
```

Software citation (CITATION.cff):
```
Lin W, Chen W, Wei G, Huang X. (2025). DeepTriage-CN (Version 1.0.0)
[Software]. Zenodo. https://doi.org/10.5281/zenodo.20073330
```

---

## 15. Contact

| Role | Person | Email |
|------|--------|-------|
| Corresponding author | Xiaolei Huang | MichaelHuangDoctor@163.com |
| Code and data queries | Wenjia Lin | *(via corresponding author)* |

For bug reports, please open an issue on GitHub. For data access requests, contact the corresponding author with a brief description of your intended use and your institutional affiliation.

---

*This repository fulfils the open science mandates of the TRIPOD+AI 2024 guidelines and has been permanently archived at Zenodo (DOI: 10.5281/zenodo.20073330).*
