#!/usr/bin/env python3
"""
generate_all_figures.py

Generates all figures from the paper:
    - Figure 2a: Overall ROC
    - Figure 2b: Geriatric ROC
    - Figure 3: Calibration
    - Figure 4: Decision Curve
    - Figure 5: SHAP beeswarm
    - Figure 6a: Error distribution
    - Figure 6b: FNR by subgroup
    - Supplementary Figure 1: Robustness bars

Uses pre-saved predictions and model outputs.

Usage:
    python scripts/generate_all_figures.py --config config.yaml
"""

import argparse
import yaml
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data, preprocess_structured, preprocess_text, get_labels,
)
from visualization.plot_roc import plot_roc_overall, plot_roc_geriatric
from visualization.plot_calibration import plot_calibration_curve
from visualization.plot_decision_curve import plot_decision_curve
from visualization.plot_robustness import plot_robustness_bars
from visualization.plot_shap import plot_shap_beeswarm
from visualization.plot_error_analysis import (
    plot_error_distribution, plot_fnr_by_subgroup,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    fig_dir = Path(config["output_paths"]["figures_dir"])
    fig_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(config["output_paths"]["models_dir"])
    results_dir = Path(config["output_paths"]["results_dir"])
    threshold = config["evaluation"]["optimal_threshold"]
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

    # Load models and predictions
    deeptriage = joblib.load(models_dir / "deeptriage_cn.pkl")
    vit_xgb = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf = joblib.load(models_dir / "random_forest.pkl")
    text_lr = joblib.load(models_dir / "text_only_lr.pkl")
    val_scores = pd.read_csv(models_dir / "clinical_scores_val.csv")

    text_enc = deeptriage.text_encoder
    val_emb = text_enc.encode(val_texts.tolist())

    preds = {
        "DeepTriage-CN": deeptriage.predict_proba(val_texts.tolist(), X_val_raw),
        "XGBoost (Vitals-only)": vit_xgb.predict_proba(X_val_raw),
        "Random Forest": rf.predict_proba(X_val_raw),
        "Text-Only (BERT+LR)": text_lr.predict_proba(val_emb),
        "NEWS2": val_scores["NEWS2"].values / 20.0,
        "MEWS": val_scores["MEWS"].values / 15.0,
        "ESI": (6 - val_scores["ESI"].values) / 5.0,
    }
    # TabNet
    from src.tabnet_model import TabNetWrapper
    tabnet = TabNetWrapper()
    tabnet.load(str(models_dir / "tabnet"))
    preds["TabNet"] = tabnet.predict_proba(X_val_raw)[:, 1]

    # 1. Overall ROC (Figure 2a)
    roc_data = {}
    for name, s in preds.items():
        fpr, tpr, _ = roc_curve(y_val, s)
        auroc = roc_auc_score(y_val, s)
        roc_data[name] = {"fpr": fpr, "tpr": tpr, "auroc": auroc}
    plot_roc_overall(roc_data, save_path=str(fig_dir / "figure2a_roc_overall.tiff"))

    # 2. Geriatric ROC (Figure 2b)
    elder_mask = val_df["age"] >= 65
    roc_ger = {}
    for name in ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)", "NEWS2", "ESI"]:
        fpr, tpr, _ = roc_curve(y_val[elder_mask], preds[name][elder_mask])
        auroc = roc_auc_score(y_val[elder_mask], preds[name][elder_mask])
        roc_ger[name] = {"fpr": fpr, "tpr": tpr, "auroc": auroc}
    plot_roc_geriatric(roc_ger, save_path=str(fig_dir / "figure2b_roc_geriatric.tiff"))

    # 3. Calibration (Figure 3)
    from evaluation.calibration import brier_score
    brier = brier_score(y_val, preds["DeepTriage-CN"])
    plot_calibration_curve(y_val, preds["DeepTriage-CN"],
                           brier_score=brier,
                           save_path=str(fig_dir / "figure3_calibration.tiff"))

    # 4. Decision curve (Figure 4)
    plot_decision_curve(
        y_val, preds["DeepTriage-CN"], news2_scores=val_scores["NEWS2"].values,
        save_path=str(fig_dir / "figure4_decision_curve.tiff")
    )

    # 5. SHAP (Figure 5)
    X_fused = np.hstack([val_emb, _scaled(X_val_raw)])  # reuse scaler
    feature_names = (
        [f"text_{i}" for i in range(768)] +
        ["age", "sex", "temp", "hr", "rr", "sbp", "dbp", "spo2"]
    )
    plot_shap_beeswarm(
        deeptriage.classifier, X_fused, feature_names,
        save_path=str(fig_dir / "figure5_shap.tiff")
    )

    # 6. Error Analysis (Figure 6)
    y_pred_bin = (preds["DeepTriage-CN"] >= threshold).astype(int)
    plot_error_distribution(
        y_val, y_pred_bin,
        save_path=str(fig_dir / "figure6a_error_distribution.tiff")
    )
    # For FNR, use the error_analysis module (requires lengths)
    comp_lens = val_df["chief_complaint"].str.len().values
    from evaluation.error_analysis import error_analysis_by_subgroup
    err_res = error_analysis_by_subgroup(
        y_val, y_pred_bin, val_df["age"].values, comp_lens,
        age_threshold=65, sparse_narrative_threshold=5,
    )
    fnr_data = {
        "young_sparse": err_res["young_sparse"]["fnr"],
        "elder_sparse": err_res["elder_sparse"]["fnr"],
        "young_rich": err_res["young_rich"]["fnr"],
        "elder_rich": err_res["elder_rich"]["fnr"],
    }
    plot_fnr_by_subgroup(fnr_data, save_path=str(fig_dir / "figure6b_fnr_subgroup.tiff"))

    # 7. Robustness bars (Supplementary Figure 1)
    rob_df = pd.read_csv(results_dir / "robustness_results.csv")
    base = dict(zip(rob_df["Model"], rob_df["Baseline_AUROC"]))
    deg = dict(zip(rob_df["Model"], rob_df["Degraded_AUROC"]))
    plot_robustness_bars(base, deg, save_path=str(fig_dir / "supp_figure1_robustness.tiff"))

    print("All figures generated and saved to:", fig_dir)


def _scaled(X_raw):
    from src.structured_encoder import StructuredEncoder
    enc = StructuredEncoder()
    return enc.fit_transform(X_raw)

if __name__ == "__main__":
    main()