#!/usr/bin/env python3
"""
robustness_simulation.py

Runs the data degradation robustness simulation (Section 3.7) and computes
performance retention for DeepTriage-CN, TabNet, and Vitals-only XGBoost.

Usage:
    python scripts/robustness_simulation.py --config config.yaml
"""

import argparse
import yaml
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    preprocess_structured,
    preprocess_text,
    get_labels,
)
from src.robustness import simulate_degraded_data
from src.structured_encoder import STRUCTURED_FEATURE_NAMES
from evaluation.metrics import compute_all_binary_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    rob_cfg = config["robustness"]
    threshold = config["evaluation"]["optimal_threshold"]
    output_dir = Path(config["output_paths"]["results_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(config["output_paths"]["models_dir"])
    random_state = config["random_state"]

    # Load data
    train_df, val_df = load_and_split_data(
        config["data"]["raw_data_path"],
        training_cutoff_date=config["data"]["training_cutoff_date"],
        validation_start_date=config["data"]["validation_start_date"],
        random_state=random_state,
    )
    X_train_raw, X_val_raw, scaler = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    # Load models
    deeptriage = joblib.load(models_dir / "deeptriage_cn.pkl")
    vit_xgb = joblib.load(models_dir / "vitals_only_xgb.pkl")
    from src.tabnet_model import TabNetWrapper
    tabnet = TabNetWrapper()
    tabnet.load(str(models_dir / "tabnet"))

    # Compute clean baseline AUROCs
    text_emb_val = deeptriage.text_encoder.encode(val_texts.tolist())
    X_val_std_clean = scaler.transform(X_val_raw)
    X_fused_clean = np.hstack([text_emb_val, X_val_std_clean])

    baseline = {}
    baseline["DeepTriage-CN"] = compute_all_binary_metrics(
        y_val, deeptriage.classifier.predict_proba(X_fused_clean)[:, 1], threshold
    )["auroc"]
    baseline["TabNet"] = compute_all_binary_metrics(
        y_val, tabnet.predict_proba(X_val_raw)[:, 1], threshold
    )["auroc"]
    baseline["XGBoost (Vitals-only)"] = compute_all_binary_metrics(
        y_val, vit_xgb.predict_proba(X_val_raw), threshold
    )["auroc"]

    # Degrade data (30% MNAR + noise)
    degraded = simulate_degraded_data(
        X_val_raw,
        STRUCTURED_FEATURE_NAMES,
        missing_proportions=[rob_cfg["missing_proportions"][-1]],  # 0.3
        noise_sigma=rob_cfg["noise_sigma_multiplier"],
        seed=random_state,
    )[rob_cfg["missing_proportions"][-1]]
    X_val_std_deg = scaler.transform(degraded)
    X_fused_deg = np.hstack([text_emb_val, X_val_std_deg])

    degraded_auroc = {}
    degraded_auroc["DeepTriage-CN"] = compute_all_binary_metrics(
        y_val, deeptriage.classifier.predict_proba(X_fused_deg)[:, 1], threshold
    )["auroc"]
    degraded_auroc["TabNet"] = compute_all_binary_metrics(
        y_val, tabnet.predict_proba(degraded)[:, 1], threshold
    )["auroc"]
    degraded_auroc["XGBoost (Vitals-only)"] = compute_all_binary_metrics(
        y_val, vit_xgb.predict_proba(degraded), threshold
    )["auroc"]

    # Retention
    retention = {k: degraded_auroc[k] / baseline[k] for k in baseline}
    results = pd.DataFrame({
        "Model": list(baseline.keys()),
        "Baseline_AUROC": list(baseline.values()),
        "Degraded_AUROC": list(degraded_auroc.values()),
        "Retention": list(retention.values()),
    })
    results.to_csv(output_dir / "robustness_results.csv", index=False)
    print(results)
    print("Robustness simulation complete. Results saved.")


if __name__ == "__main__":
    main()