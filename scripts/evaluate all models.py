#!/usr/bin/env python3
"""
evaluate_all_models.py

Evaluates all trained models on the temporal validation set.
Generates:
    - Table 2: AUROC, AUPRC, sensitivity, specificity, accuracy
    - Calibration metrics (Brier score, intercept/slope, Hosmer-Lemeshow)
    - Bootstrap confidence intervals
    - DeLong tests and continuous NRI
Saves results to outputs/results/.

Usage:
    python scripts/evaluate_all_models.py --config config.yaml
"""

import argparse
import yaml
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    preprocess_structured,
    preprocess_text,
    get_labels,
)
from evaluation.metrics import compute_all_binary_metrics
from evaluation.calibration import compute_calibration_metrics
from evaluation.bootstrap import bootstrap_confidence_intervals
from evaluation.statistical_tests import delong_test, continuous_nri


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    threshold = config["evaluation"]["optimal_threshold"]
    n_bootstrap = config["evaluation"]["n_bootstrap"]
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
    X_train_raw, X_val_raw, _ = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    # Load models
    deeptriage = joblib.load(models_dir / "deeptriage_cn.pkl")
    vit_xgb = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf = joblib.load(models_dir / "random_forest.pkl")
    text_lr = joblib.load(models_dir / "text_only_lr.pkl")
    val_scores = pd.read_csv(models_dir / "clinical_scores_val.csv")

    # Generate predictions
    text_enc = deeptriage.text_encoder
    val_emb = text_enc.encode(val_texts.tolist())

    # TabNet predictions
    from src.tabnet_model import TabNetWrapper
    tabnet = TabNetWrapper()
    tabnet.load(str(models_dir / "tabnet"))
    tabnet_proba = tabnet.predict_proba(X_val_raw)[:, 1]

    preds = {
        "DeepTriage-CN": deeptriage.predict_proba(val_texts.tolist(), X_val_raw),
        "TabNet": tabnet_proba,
        "XGBoost (Vitals-only)": vit_xgb.predict_proba(X_val_raw),
        "Random Forest": rf.predict_proba(X_val_raw),
        "Text-Only (BERT+LR)": text_lr.predict_proba(val_emb),
        "NEWS2": val_scores["NEWS2"].values / 20.0,
        "MEWS": val_scores["MEWS"].values / 15.0,
        "ESI": (6 - val_scores["ESI"].values) / 5.0,  # invert
    }

    # Compute metrics (Table 2)
    results = []
    for name, scores in preds.items():
        metrics = compute_all_binary_metrics(y_val, scores, threshold)
        metrics["Model"] = name
        results.append(metrics)
    results_df = pd.DataFrame(results).set_index("Model")
    results_df.round(3).to_csv(output_dir / "table2_metrics.csv")
    print("Table 2 saved.")

    # Bootstrap CIs for main models
    main_models = ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)", "NEWS2", "ESI"]
    ci_dict = {}
    for name in main_models:
        ci = bootstrap_confidence_intervals(
            y_val, preds[name], n_bootstrap=n_bootstrap, random_seed=random_state
        )
        ci_dict[name] = ci
    pd.DataFrame(ci_dict).T.to_csv(output_dir / "bootstrap_cis.csv")
    print("Bootstrap CIs saved.")

    # DeLong tests
    with open(output_dir / "delong_tests.txt", "w") as f:
        for name in ["TabNet", "XGBoost (Vitals-only)", "NEWS2", "ESI"]:
            z, p = delong_test(y_val, preds["DeepTriage-CN"], preds[name])
            f.write(f"DeepTriage-CN vs {name}: z={z:.4f}, p={p:.4f}\n")
    print("DeLong tests saved.")

    # Continuous NRI
    nri_news2 = continuous_nri(y_val, preds["NEWS2"], preds["DeepTriage-CN"])
    nri_esi = continuous_nri(y_val, preds["ESI"], preds["DeepTriage-CN"])
    nri_df = pd.DataFrame({
        "Comparison": ["vs NEWS2", "vs ESI"],
        "NRI_total": [nri_news2[0], nri_esi[0]],
        "NRI_events": [nri_news2[1], nri_esi[1]],
        "NRI_nonevents": [nri_news2[2], nri_esi[2]],
    })
    nri_df.to_csv(output_dir / "nri_results.csv", index=False)
    print("NRI results saved.")

    # Calibration
    cal = compute_calibration_metrics(y_val, preds["DeepTriage-CN"])
    pd.Series(cal).to_csv(output_dir / "calibration_metrics.csv")
    print("Calibration metrics saved.")

    print("Evaluation complete.")


if __name__ == "__main__":
    main()