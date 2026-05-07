"""
DeepTriage-CN Source Package
=============================
Core modules for preprocessing, model architectures, and robustness testing.

Modules:
    preprocess          - Data loading, cleaning, and train/val splitting
    text_encoder        - Frozen BERT-Chinese feature extraction
    structured_encoder  - Standardization and feature concatenation for tabular data
    fusion_model        - Late-fusion XGBoost classifier (the DeepTriage-CN model)
    tabnet_model        - TabNet implementation with paper-specified hyperparameters
    baseline_models     - XGBoost (vitals-only), Random Forest, Logistic Regression (text-only)
    clinical_scores     - NEWS2, MEWS, ESI scoring calculators
    robustness          - MNAR missingness simulation and noise injection

Author: Wenjia Lin, Wenliang Chen, Guan Wei, Xiaolei Huang
Affiliation: The Second Affiliated Hospital of Fujian Medical University
"""

__version__ = "1.0.0"
__all__ = [
    "preprocess",
    "text_encoder",
    "structured_encoder",
    "fusion_model",
    "tabnet_model",
    "baseline_models",
    "clinical_scores",
    "robustness",
]