```markdown
# DeepTriage-CN: Multimodal Emergency Department Admission Prediction

[![DOI](<img width="191" height="20" alt="image" src="https://github.com/user-attachments/assets/0b5cbc2e-1856-4c7d-8bfc-3ca3b15a3445" />
](DOI：10.5281/zenodo.20073330)

This repository contains the official implementation of **DeepTriage-CN**, a late-fusion multimodal framework for predicting hospital admission from information available at emergency department (ED) triage. It integrates frozen BERT-Chinese embeddings of nurse-recorded chief complaints with structured vital signs using an XGBoost classifier.

## Key Findings (from our paper)

- **Overall Performance:** Achieved an AUROC of **0.865** on a temporal validation set (N=2,000), significantly outperforming traditional clinical scores (NEWS2: 0.772; ESI: 0.760).
- **Comparable to TabNet:** Performance is on par with an optimized TabNet deep learning model (AUROC 0.867) trained on tabular data alone.
- **Robustness in Older Adults:** Maintained robust discriminative capacity in patients aged ≥65 (AUROC 0.852), where conventional scores declined markedly.
- **Stability Under Data Degradation:** Demonstrated superior resilience, retaining **95.4%** of baseline performance under simulated 30% missing-not-at-random (MNAR) conditions with noise.

Our work suggests that while unstructured text may not always provide incremental discriminative gains over highly tuned tabular deep learners, it offers a stabilizing anchor when physiological signals are degraded or less informative.

## System Architecture

![DeepTriage-CN Architecture](outputs/figures/figure1_architecture.png)

The `DeepTriage-CN` framework implements a two-tower design:
1.  **Text Tower:** A frozen `bert-base-chinese` model extracts a 768-dimensional `[CLS]` embedding from the chief complaint.
2.  **Structured Tower:** 8 numerical features (age, sex, 6 vital signs) are standardized.
3.  **Fusion & Classification:** The 776-dimensional fused vector is input to an XGBoost classifier for final prediction.

## Repository Structure

```text
DeepTriage-CN/
├── data/               # Instructions to generate/use synthetic data
├── src/                # Core source code for models and preprocessing
├── evaluation/         # Metrics, calibration, decision curve, bootstrap, error analysis
├── visualization/      # Scripts to reproduce all figures from the paper
├── notebooks/          # Step-by-step Jupyter notebooks for exploration and analysis
├── scripts/            # High-level scripts to run the full pipeline
├── outputs/            # Trained model weights, results tables, and generated figures
├── tests/              # Unit tests for critical components
└── docs/               # Additional documentation
```

## Hardware Requirements

- **Minimum:** 8GB RAM, modern multi-core CPU. 
- **Recommended:** 16GB+ RAM. A GPU (e.g., NVIDIA RTX 3060 or higher) is highly recommended if you plan to re-extract `[CLS]` embeddings from large text datasets using `bert-base-chinese`. 
- **Estimated Time:** Training the late-fusion XGBoost model on the 8,000-sample dataset takes less than 2 minutes on a standard CPU. However, generating BERT embeddings for 10,000 texts takes approximately 10-15 minutes on a standard GPU, or up to an hour on a CPU.

## Environment Setup

We recommend using Conda to manage the environment.

**Option 1: Using `environment.yml` (Recommended)**
```bash
conda env create -f environment.yml
conda activate deeptriage-cn
```

**Option 2: Using `requirements.txt` (For pip users)**
```bash
pip install -r requirements.txt
```
*Note: `pytorch` installation varies by system. Please install it according to the official guide if using `requirements.txt`.*

## Quick Start: Reproducing Main Results

1.  **Generate Synthetic Data:** (Since real patient data cannot be shared due to institutional ethics)
    ```bash
    python data/generate_synthetic_data.py
    ```

2.  **Run the Full Pipeline:** This single script executes preprocessing, training, evaluation, and figure generation.
    ```bash
    python scripts/run_full_pipeline.py --config config.yaml
    ```
    All outputs (models, metrics, figures) will be saved to the `outputs/` directory.

3.  **Explore via Notebooks:** For a more detailed, step-by-step walkthrough, use the Jupyter notebooks in the `notebooks/` directory.

## Custom Data Formatting

If you wish to evaluate DeepTriage-CN on your own external validation dataset, your input CSV must strictly contain the following 9 columns (8 structured + 1 unstructured):

| Feature Name | Type | Description |
| :--- | :--- | :--- |
| `age` | Integer | Patient age in years (≥18) |
| `sex` | Integer | Binary: 0 (Female), 1 (Male) |
| `temperature` | Float | Tympanic/oral temperature in °C |
| `heart_rate` | Integer | Beats per minute |
| `respiratory_rate`| Integer | Breaths per minute |
| `systolic_bp` | Integer | Systolic blood pressure in mmHg |
| `diastolic_bp` | Integer | Diastolic blood pressure in mmHg |
| `spo2` | Integer | Peripheral oxygen saturation (%) |
| `chief_complaint` | String | Raw text recorded at triage. Use `[MISSING]` if unavailable. |
| `admission` | Integer | Target label: 0 (Discharged), 1 (Admitted) |

## Downloading Pre-trained Weights

To skip the training pipeline and run inference directly:
1. Download the pre-trained model weights (`deeptriage_xgb_fusion.pkl` and `tabnet_optimal.zip`) from our [Zenodo release](https://doi.org/10.5281/zenodo.XXXXXXX).
2. Place the downloaded files into the `outputs/models/` directory.
3. You can now use the `predict_single_patient.py` script without running the full training loop.

## Using a Single-Patient Prediction Script

After configuring the weights, you can run inference on a single patient to demonstrate the pipeline's real-time workflow:
```bash
python scripts/inference_single_patient.py --complaint "胸痛并伴有呼吸急促" --age 75 --heart_rate 110 --respiratory_rate 26 --spo2 93
```

## Maintaining the Integrity of Findings

We designed this repository to ensure computational reproducibility. Key configurations (e.g., random seeds, model hyperparameters, data split dates) are explicitly set in `config.yaml` and throughout the code to match our paper's findings. Please refer to `docs/reproducibility_guide.md` for a detailed checklist.

## Citation

If you use this code or model in your research, please cite our paper:
```bibtex
@article{lin2025deeptriage,
  title={DeepTriage-CN: Integrating Clinical Text with Vital Signs for Emergency Department Admission Prediction in an Aging Population},
  author={Lin, Wenjia and Chen, Wenliang and Wei, Guan and Huang, Xiaolei},
  journal={...},
  year={2025},
  publisher={...}
}
```
Please also check the `CITATION.cff` file for structured metadata.

## License

This project is licensed under the MIT License - see the `LICENSE` file for details.

## Disclaimer

This tool is for research purposes only. It is a non-interventional, associative descriptive model and is **not** intended to replace clinical judgment or correct triage decisions. Any prospective clinical use would require rigorous validation and ethical approval.
```
