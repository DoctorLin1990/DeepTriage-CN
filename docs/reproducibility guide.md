markdown
# Reproducibility Guide for DeepTriage-CN

This document provides **step-by-step instructions** to computationally reproduce the results reported in:

> **DeepTriage-CN: Integrating Clinical Text with Vital Signs for Emergency Department Admission Prediction in an Aging Population**  
> Wenjia Lin, Wenliang Chen, Guan Wei, Xiaolei Huang

---

## 1. System Requirements

- **Operating System**: Linux (recommended), macOS, or Windows with WSL2
- **Python**: 3.9
- **CUDA**: Optional – all scripts run on CPU; a GPU is recommended for faster BERT inference and TabNet training
- **Conda** (recommended) or `pip` for environment management

---

## 2. Environment Setup

### Option A – Conda (Recommended)

```bash
cd DeepTriage-CN
conda env create -f environment.yml
conda activate deeptriage-cn
Option B – pip + venv
bash
python3.9 -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
# Install PyTorch separately according to your CUDA version:
# pip install torch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 --index-url https://download.pytorch.org/whl/cpu
Verify installation:

bash
python -c "import torch; import xgboost; import transformers; print('Environment OK')"
3. Data
Due to institutional privacy regulations, real EHR data cannot be publicly shared. We provide a synthetic data generator that creates a dataset with the same schema and statistical characteristics (distributions, missingness proportions, age-specific admission rates) as described in the paper.

Generate Synthetic Dataset
bash
python data/generate_synthetic_data.py
Output: data/sample_data.csv (10,000 encounters)

Important caveat: Models trained on synthetic data will produce evaluation metrics that approximate but do not exactly replicate the paper’s reported AUROCs. Exact reproduction requires access to the original de‑identified dataset, which can be requested from the corresponding author subject to IRB approval.

4. Reproducing Results – Quick Start
The entire pipeline can be executed with a single command:

bash
python scripts/run_full_pipeline.py --config config.yaml
This will sequentially:

Train all models (DeepTriage-CN, TabNet, XGBoost, Random Forest, Text‑Only LR)

Evaluate them on the temporal validation set → generates Table 2, calibration metrics, bootstrap CIs, DeLong tests, NRI

Run robustness simulation → Supplementary Figure 1 data

Generate all figures → outputs/figures/

Outputs are written to outputs/ with the following structure:

text
outputs/
├── models/                # Trained model weights
│   ├── deeptriage_cn.pkl
│   ├── tabnet*            (TabNet files)
│   ├── vitals_only_xgb.pkl
│   ├── random_forest.pkl
│   ├── text_only_lr.pkl
│   └── clinical_scores_val.csv
├── results/               # Evaluation metrics tables
│   ├── table2_metrics.csv
│   ├── bootstrap_cis.csv
│   ├── delong_tests.txt
│   ├── nri_results.csv
│   ├── calibration_metrics.csv
│   └── robustness_results.csv
└── figures/               # TIFF figures (600 dpi)
    ├── figure2a_roc_overall.tiff
    ├── figure2b_roc_geriatric.tiff
    ├── figure3_calibration.tiff
    ├── figure4_decision_curve.tiff
    ├── figure5_shap.tiff
    ├── figure6a_error_distribution.tiff
    ├── figure6b_fnr_subgroup.tiff
    └── supp_figure1_robustness.tiff
5. Step‑by‑Step Manual Execution
If you prefer to run each phase separately:

5.1 Training
bash
python scripts/train_all_models.py --config config.yaml
5.2 Evaluation
bash
python scripts/evaluate_all_models.py --config config.yaml
5.3 Robustness Simulation
bash
python scripts/robustness_simulation.py --config config.yaml
5.4 Figures
bash
python scripts/generate_all_figures.py --config config.yaml
All scripts accept --config to specify a YAML configuration file (default: config.yaml).

6. Hyperparameter Tuning (Optional)
The models in the paper were trained with hyperparameters obtained via grid search (Section 3.6). These optimal values are already set in config.yaml. To re‑run tuning:

bash
python scripts/tune_hyperparameters.py --config config.yaml --model tabnet
python scripts/tune_hyperparameters.py --config config.yaml --model xgboost
Note: Tuning is time‑consuming and not required for reproducing the paper’s main results.

7. Running Tests
Unit tests ensure the core modules are functioning correctly:

bash
pytest tests/ -v
8. Using Jupyter Notebooks
For interactive exploration, start Jupyter and run the notebooks in order:

bash
jupyter notebook notebooks/01_data_exploration.ipynb
Notebooks are numbered and designed to be executed sequentially.

9. Single‑Patient Prediction Demo
After training, you can test inference on an individual case:

bash
python scripts/predict_single_patient.py \
    --config config.yaml \
    --complaint "胸痛并伴有呼吸急促" \
    --age 75 --heart_rate 110 --respiratory_rate 26 \
    --spo2 93 --temperature 37.0 --sbp 145 --dbp 88 --sex 1
10. Notes on Exact Numerical Reproducibility
Random seeds are fixed in config.yaml (random_state: 42). All training‑validation splits, model initialisations, and bootstrap resampling use this seed.

PyTorch determinism: Due to non‑deterministic operations in transformer attention, BERT embeddings may vary slightly across hardware (CPU vs GPU). This can cause minor variations in AUROC (typically <0.002) but does not affect conclusions.

TabNet training involves stochastic gradient descent; results may vary within a small range. The paper’s values are the mean of three runs; for exact replication use the provided model weights if available.

11. Contact
For questions about reproducibility or data access, please contact the corresponding author:
Xiaolei Huang – MichaelHuangDoctor@163.com

*Last updated: 2025-05-07*
Document version: 1.0.0