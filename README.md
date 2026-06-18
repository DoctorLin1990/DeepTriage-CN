# DeepTriage-CN

**DeepTriage-CN: Integrating Clinical Text with Vital Signs for Emergency Department Admission Prediction in an Aging Population**

*Scientific Reports* · Wenjia Lin, Wenliang Chen, Guan Wei, Xiaolei Huang

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20073330.svg)](https://doi.org/10.5281/zenodo.20073330)

---

## Overview

DeepTriage-CN is a **late-fusion multimodal framework** for predicting hospital admission from emergency department triage data. It combines:

- A **frozen bert-base-chinese encoder** that extracts 768-dimensional semantic embeddings from nurse-recorded chief complaints
- **Eight standardised triage vital signs** (temperature, heart rate, respiratory rate, systolic BP, diastolic BP, SpO₂, age, sex)
- An **XGBoost classifier** trained on the concatenated 776-dimensional feature vector

The framework was developed and validated on 10,000 adult ED encounters (8,000 training / 2,000 temporal validation) at The Second Affiliated Hospital of Fujian Medical University, Quanzhou, Fujian, China (January 2023 – March 2024).

### Key Results (Temporal Validation Set, N = 2,000)

| Model | AUROC (95% CI) | Youden Threshold | PPV | Alert Rate |
|-------|---------------|-----------------|-----|------------|
| **DeepTriage-CN** | **0.865 (0.848–0.879)** | 0.28 | **0.50** | 22.4% |
| TabNet | 0.867 (0.852–0.881) | — | — | — |
| XGBoost (Vitals-only) | 0.858 (0.841–0.872) | — | — | — |
| NEWS2 | 0.772 (0.751–0.792) | — | — | — |
| ESI | 0.760 (0.741–0.782) | — | — | — |

> **Primary finding**: In the general ED population, adding frozen BERT embeddings of nurse-recorded chief complaints does not confer statistically significant discriminative improvement over an optimised tabular deep-learning model (TabNet; DeLong *p* = 0.42). The multimodal approach offers two bounded situational advantages: (1) differential robustness under MNAR data degradation (AUROC retention 95.4% vs 83.3% for TabNet at 30% missingness) and (2) significantly amplified advantage over conventional clinical scoring instruments in geriatric patients (age × model interaction *p* = 0.001 vs NEWS2).

---

## Repository Structure

```
DeepTriage-CN/
├── config.yaml                    # Single source of truth for all parameters
├── requirements.txt               # Python dependencies
├── environment.yaml               # Conda environment specification
├── LICENSE
│
├── src/                           # Core model modules
│   ├── __init__.py
│   ├── text_encoder.py            # Frozen BERT-Chinese feature extractor
│   ├── structured_encoder.py      # NaN-safe z-score normaliser
│   ├── fusion_model.py            # DeepTriageCN (two-tower late fusion)
│   ├── tabnet_model.py            # TabNet wrapper (benchmark)
│   ├── baseline_models.py         # XGBoost / Random Forest / Text-only LR
│   ├── clinical_scores.py         # NEWS2 / MEWS / ESI computation
│   └── preprocess.py              # Data loading and temporal split
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py                 # AUROC, DeLong, HL, NRI, alert burden
│   └── decision_curve.py          # Net benefit computation for DCA
│
├── visualization/
│   ├── __init__.py
│   ├── generate_figures_nature.py # Figure 2, 3, 4 (Nature style, 300 DPI TIFF)
│   ├── compute_shap.py            # Real SHAP computation via TreeExplainer
│   └── plot_robustness.py         # Supplementary Figure S1
│
├── scripts/
│   ├── train_all_models.py        # Train DeepTriage-CN + all baselines
│   ├── evaluate_all_models.py     # Tables 2, 3, 4 + interaction contrasts
│   ├── robustness_simulation.py   # MNAR simulation (Section 3.7)
│   └── generate_all_figures.py    # End-to-end figure generation
│
├── data/
│   ├── preprocess_real_data.py    # Validate & clean institutional dataset
│   └── feature_names.json         # Feature schema with paper statistics
│
├── tests/
│   └── test_pipeline.py           # Unit + integration tests (pytest)
│
└── outputs/                       # Auto-created at runtime
    ├── models/
    ├── results/
    ├── figures/
    └── logs/
```

---

## Installation

### Option A: Conda (recommended)

```bash
git clone https://github.com/MichaelHuangDoctor/DeepTriage-CN.git
cd DeepTriage-CN
conda env create -f environment.yaml
conda activate deeptriage-cn
```

### Option B: pip

```bash
git clone https://github.com/MichaelHuangDoctor/DeepTriage-CN.git
cd DeepTriage-CN
pip install -r requirements.txt
```

---

## Reproducibility

### Step 0 — Prepare your data

```bash
python data/preprocess_real_data.py \
    --input  /path/to/institutional_ed_data.csv \
    --output data/ed_visits.csv \
    --config config.yaml
```

The expected column format is described in `data/feature_names.json`.  
All column names are configurable via `config.yaml → column_mapping`.

### Step 1 — Train all models

```bash
python scripts/train_all_models.py --config config.yaml
```

This trains DeepTriage-CN, TabNet, XGBoost (Vitals-only), Random Forest,
Text-only LR, and computes NEWS2 / MEWS / ESI scores on the validation set.
All artefacts are saved to `outputs/models/`.

Training time (approximate):
- bert-base-chinese encoding (CPU): ~15 min for 8,000 texts
- bert-base-chinese encoding (GPU): ~2 min
- XGBoost classifier: ~30 sec

### Step 2 — Evaluate

```bash
python scripts/evaluate_all_models.py --config config.yaml
```

Outputs: `outputs/results/table2_performance.csv`, `table3_geriatric_calibration.csv`,
`table4_alert_burden.csv`, `interaction_contrasts.csv`, `nri_results.csv`.

### Step 3 — Robustness simulation

```bash
python scripts/robustness_simulation.py --config config.yaml
```

Simulates MNAR missingness at 10%, 20%, 30% with Gaussian noise (Section 3.7).

### Step 4 — Compute SHAP values

```bash
python visualization/compute_shap.py \
    --train data/ed_visits.csv \
    --test  data/ed_visits.csv \
    --model outputs/models/deeptriage_cn.pkl \
    --outdir outputs/results \
    --shap_n 500 \
    --save_shap
```

### Step 5 — Generate all figures

```bash
python scripts/generate_all_figures.py --config config.yaml
```

Produces `Figure2_ROC.tiff`, `Figure3_Calibration_DCA.tiff`,
`Figure4_SHAP_Error_FNR.tiff`, `SuppFigure1_Robustness.tiff`
in `outputs/figures/`.

---

## Configuration

All hyperparameters, file paths, and thresholds are centralised in `config.yaml`.
**No hard-coded values appear in the source code.**

Key parameters (matching the paper):

```yaml
xgboost:
  n_estimators: 200
  learning_rate: 0.05
  max_depth: 6
  subsample: 0.8
  colsample_bytree: 0.8

text_processing:
  model_name: bert-base-chinese
  max_length: 64

evaluation:
  optimal_threshold: 0.28    # Youden-optimal (Section 4.2)
  n_bootstrap: 1000
```

---

## Data Availability

De-identified patient data cannot be shared publicly due to institutional
privacy regulations (Ethics Approval: FMU-AH-2023-699). Requests for
access to anonymised aggregated data may be directed to the corresponding
author subject to institutional review board approval.

All custom code, model architectures, and hyperparameter configurations
are archived at Zenodo (DOI: [10.5281/zenodo.20073330](https://doi.org/10.5281/zenodo.20073330)).

---

## Ethical Statement

This study was approved by the Institutional Review Board of The Second
Affiliated Hospital of Fujian Medical University (Approval No: FMU-AH-2023-699).
The requirement for informed consent was waived because the analysis used
de-identified retrospective administrative data. All procedures complied with
the Declaration of Helsinki.

**Limitations**: Hospital admission is confounded by bed availability, ED census,
and physician risk tolerance. The model reflects operational practice, not
purely physiological acuity. At the Youden-optimal threshold (0.28), PPV is 0.50,
meaning one in two alerts is a false positive. Prospective multicentre validation
is required before clinical deployment.

---

## Citation

```bibtex
@article{lin2025deeptriage,
  title   = {DeepTriage-CN: Integrating Clinical Text with Vital Signs for
             Emergency Department Admission Prediction in an Aging Population},
  author  = {Lin, Wenjia and Chen, Wenliang and Wei, Guan and Huang, Xiaolei},
  journal = {Scientific Reports},
  year    = {2025},
  doi     = {10.1038/s41598-025-XXXXX-X}
}
```

---

## Funding

Fujian Medical University QiHang Fund (Grant No. 2023QH1130).

## License

MIT License — see [LICENSE](LICENSE).

## Contact

Corresponding author: Xiaolei Huang  
Department of Emergency Medicine  
The Second Affiliated Hospital of Fujian Medical University  
Quanzhou, Fujian, China  
Email: MichaelHuangDoctor@163.com
