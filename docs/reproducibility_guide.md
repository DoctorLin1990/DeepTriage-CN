# Reproducibility Guide — DeepTriage-CN

This document provides a detailed step-by-step guide for reproducing every
quantitative result, table, and figure reported in the paper.

---

## 0. Prerequisites

### Hardware

| Component | Requirement |
|-----------|-------------|
| RAM | ≥ 8 GB |
| GPU | Optional but recommended (BERT encoding, TabNet training). CPU-only is fully supported. |
| Disk space | ≥ 10 GB (5 GB for `bert-base-chinese` weights, rest for data/outputs). |

### Python environment

```bash
# Conda (recommended)
conda env create -f environment.yaml
conda activate deeptriage

# OR pip
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

---

## 1. Data Preparation

### Option A — Synthetic dataset (no patient data required)

```bash
python data/generate_synthetic_data.py
# → writes data/synthetic_ed_visits.csv  (~1 MB)
```

The synthetic dataset replicates:
- 10,000 encounters (8,000 training / 2,000 validation)
- Admission rate ≈ 22%
- Age distribution: median 48, IQR 34–62, ≈ 20.5% geriatric (≥65)
- Per-feature missing rates from Table 1 (SpO₂ 8.2%, RR 5.2%, etc.)
- Chief complaints drawn from a fixed 20-phrase Mandarin vocabulary

**Expected runtime:** < 5 seconds.

> Results from synthetic data will NOT match the paper's reported values.
> The synthetic pipeline validates code correctness, not result magnitudes.

### Option B — Real institutional dataset

1. Ensure your CSV contains the following columns (matching `data/feature_names.json`):

   ```
   visit_date, patient_id, age, sex, temperature, heart_rate,
   respiratory_rate, sbp, dbp, spo2, chief_complaint, hospital_admission
   ```

2. Update `config.yaml`:
   ```yaml
   data:
     raw_data_path: "/path/to/your/real_data.csv"
   ```

3. For per-patient deduplication (first ED visit only, Section 3.2), uncomment
   lines 70–73 in `src/preprocess.py`:
   ```python
   df = df.sort_values("visit_date").drop_duplicates(
       subset="patient_id", keep="first"
   )
   ```

---

## 2. Training All Models

```bash
python scripts/train_all_models.py --config config.yaml
```

### What this does

| Sub-step | Description | Expected time (GPU) | Expected time (CPU) |
|----------|-------------|--------------------|--------------------|
| BERT encoding (train) | 768-d embeddings for 8,000 complaints | ~3 min | ~25 min |
| DeepTriage-CN XGBoost | Fit on 776-d fused features | < 1 min | < 1 min |
| TabNet | 200 epochs, early stopping | ~8 min | ~40 min |
| VitalsXGB | Standard XGBoost training | < 30 s | < 30 s |
| Random Forest | 200 trees | < 30 s | < 30 s |
| TextOnly LR | Logistic Regression on BERT embeddings | < 30 s | < 30 s |
| Clinical scores | NEWS2, MEWS, ESI on validation set | < 10 s | < 10 s |

### Outputs (`outputs/models/`)

```
deeptriage_cn.pkl          ← DeepTriage-CN full state dict
tabnet/                    ← TabNet model directory
vitals_only_xgb.pkl
random_forest.pkl
text_only_lr.pkl
text_encoder.pkl           ← Frozen BERT encoder (reused in evaluation)
clinical_scores_val.csv    ← Pre-computed NEWS2/MEWS/ESI for val set
```

### Key implementation notes (bug fixes)

- **L1/L2 fix**: `DeepTriageCN.fit()` receives **raw** arrays (`get_raw_structured()`); the model applies its own `StructuredEncoder` internally. `TabNetWrapper.fit()` also receives raw arrays (Ghost BN handles normalisation). Only `VitalsOnlyXGBoost`, `RandomForestBaseline`, and `TextOnlyLogisticRegression` receive pre-standardised arrays from `preprocess_structured()`.
- **B3/B4 fix**: `DeepTriageCN.save()` serialises the full model state. `DeepTriageCN.load_from_file()` reconstructs a ready instance with `predict_proba()` available.

---

## 3. Evaluating All Models (Table 2)

```bash
python scripts/evaluate_all_models.py --config config.yaml
```

### Outputs (`outputs/results/`)

| File | Contents | Paper location |
|------|----------|---------------|
| `table2_metrics.csv` | AUROC, AUPRC, sensitivity, specificity, accuracy, F1, PPV, NPV for all 8 models | Table 2 |
| `bootstrap_cis.csv` | 95% CIs from 1,000 bootstrap resamples | Table 2, footnote |
| `delong_tests.csv` | DeLong z-statistic and p-value — DeepTriage-CN vs each comparator | Section 4.2 |
| `nri_results.csv` | Continuous NRI (DeepTriage-CN vs NEWS2 and ESI) | Section 4.2 |
| `calibration_metrics.csv` | Brier score, slope, intercept, Hosmer–Lemeshow | Section 4.3 |

### Expected paper values (real data)

| Model | AUROC |
|-------|-------|
| DeepTriage-CN | 0.865 (95% CI 0.848–0.879) |
| TabNet | 0.867 (0.852–0.881) |
| XGBoost (Vitals-only) | 0.858 (0.841–0.872) |
| NEWS2 | 0.772 (0.751–0.792) |
| ESI | 0.760 (0.741–0.782) |

DeLong p-values (vs DeepTriage-CN): TabNet p=0.42, XGBoost p=0.31, NEWS2 p<0.001.

Calibration: Brier=0.12, slope=0.95, intercept=0.02, HL p=0.28.

---

## 4. Robustness Simulation (Section 3.7, Supplementary Figure 1)

```bash
python scripts/robustness_simulation.py --config config.yaml
```

### What this does

Applies MNAR missingness simulation at three levels (10%, 20%, 30%) plus
Gaussian noise (σ = 0.5 × per-feature SD) to the clean validation set, then
re-evaluates all three structural models.

**L3 fix**: all three levels are iterated (original code only ran 30%).

### Outputs

| File | Contents |
|------|----------|
| `robustness_results.csv` | Full 3-model × 3-level table with Baseline_AUROC, Degraded_AUROC, Retention_pct |
| `robustness_summary.txt` | Human-readable 30% level summary |

### Expected paper values (30% level)

| Model | Baseline | Degraded | Retention |
|-------|----------|----------|-----------|
| DeepTriage-CN | 0.865 | 0.825 | 95.4% |
| TabNet | 0.867 | 0.722 | 83.2% |
| XGBoost | 0.858 | 0.710 | 82.8% |

---

## 5. Generating All Figures

```bash
python scripts/generate_all_figures.py --config config.yaml
```

**Requires**: completed steps 2, 3, and 4 (models + robustness results).

### Outputs (`outputs/figures/`) — 300 DPI TIFF

| File | Figure | Paper section |
|------|--------|--------------|
| `figure2a_roc_overall.tiff` | Fig. 2a | Section 4.1 |
| `figure2b_roc_geriatric.tiff` | Fig. 2b | Section 4.4 |
| `figure3_calibration.tiff` | Fig. 3 | Section 4.3 |
| `figure4_decision_curve.tiff` | Fig. 4 | Section 4.3 |
| `figure5_shap.tiff` | Fig. 5 | Section 4.6 |
| `figure6a_error_distribution.tiff` | Fig. 6a | Section 4.7 |
| `figure6b_fnr_subgroup.tiff` | Fig. 6b | Section 4.7 |
| `supp_figure1_robustness.tiff` | Supp. Fig. 1 | Section 4.5 |

### Bug fixes in figure generation

- **L4 fix**: SHAP uses `deeptriage.structured_encoder.transform(X_val_raw)` — the model's own fitted encoder — instead of creating a fresh encoder fitted on validation data (data leakage).
- **L5 fix**: Complaint length for Figure 6b uses word count (`str.split().str.len()`), not character count.
- **B5 fix**: `fnr_data` values passed to `plot_fnr_by_subgroup` are `(fnr_float, n_int)` tuples (required for Wilson CI computation).

---

## 6. Running Smoke Tests

```bash
pytest tests/test_smoke.py -v
```

All 17 tests run without GPU and complete in approximately 60–90 seconds. They validate:

- `StructuredEncoder`: NaN passthrough, no data leakage between fit/transform
- `clinical_scores`: B1 walrus-operator fix (no `SyntaxError`), shape correctness
- `preprocess`: raw vs standardised arrays differ as expected
- `metrics`: all metric values in valid [0, 1] range
- `statistical_tests`: DeLong z=0 for identical predictions; NRI in [-2, 2]
- `robustness`: all 3 proportions returned (L3 fix), NaN rate monotone
- `decision_curve`: net benefit non-negative
- `error_analysis`: correct subgroup keys and FNR bounds

---

## 7. One-Command Pipeline

```bash
# Full pipeline (synthetic data)
python scripts/run_full_pipeline.py --config config.yaml

# Skip data generation (synthetic CSV already exists)
python scripts/run_full_pipeline.py --config config.yaml --skip-datagen

# Skip figure generation (faster CI)
python scripts/run_full_pipeline.py --config config.yaml --skip-figures
```

**B6 fix**: All script names use underscores (original had spaces, breaking `subprocess` calls).

---

## 8. Reproducing Individual Tables / Figures

To reproduce a specific result without running the full pipeline:

```python
import sys; sys.path.insert(0, ".")
import yaml, numpy as np, pandas as pd, joblib

config = yaml.safe_load(open("config.yaml"))

# Load models
from src.fusion_model import DeepTriageCN
from src.preprocess import load_and_split_data, get_raw_structured, preprocess_text, get_labels

train_df, val_df = load_and_split_data(config["data"]["raw_data_path"])
X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)
_, val_texts = preprocess_text(train_df, val_df)
y_train, y_val = get_labels(train_df, val_df)

deeptriage = DeepTriageCN.load_from_file("outputs/models/deeptriage_cn.pkl")
proba = deeptriage.predict_proba(val_texts.tolist(), X_val_raw)

# AUROC
from evaluation.metrics import compute_all_binary_metrics
metrics = compute_all_binary_metrics(y_val, proba, threshold=0.28)
print(f"AUROC = {metrics['auroc']:.4f}")
```

---

## 9. Frequently Asked Questions

**Q: The BERT weights take too long to download on first run.**
A: Run `python -c "from transformers import BertModel; BertModel.from_pretrained('bert-base-chinese')"` once with a stable internet connection. The weights (~400 MB) are then cached in `~/.cache/huggingface/`.

**Q: I get `CUDA out of memory` during TabNet training.**
A: Reduce `tabnet.batch_size` in `config.yaml` (e.g. from 256 to 64).

**Q: Metric values differ from the paper.**
A: This is expected when using the synthetic dataset. The paper reports values from real, undisclosed patient data. Synthetic values will differ substantially.

**Q: Can I use my own BERT model (e.g. a domain-adapted clinical Chinese BERT)?**
A: Yes. Update `config.yaml → text_processing.model_name` with any HuggingFace model identifier compatible with `BertTokenizer` and `BertModel`. Set `max_length` appropriately for the new vocabulary.

**Q: How do I request the real dataset?**
A: Contact the corresponding author at MichaelHuangDoctor@163.com with your institutional affiliation and intended use. A formal Data Use Agreement is required.
