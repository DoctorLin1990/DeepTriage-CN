#!/usr/bin/env python3
"""
generate_all_figures.py

Generates all paper figures and saves them as 300-dpi TIFF files:

    Figure 2a  – Overall ROC curve comparison
    Figure 2b  – Geriatric-subgroup ROC curve comparison
    Figure 3   – Calibration plot with density sub-panel
    Figure 4   – Decision Curve Analysis
    Figure 5   – SHAP beeswarm summary
    Figure 6a  – Misclassification distribution
    Figure 6b  – False-negative rate by age × narrative-length subgroup
    Supp. Fig. 1 – Robustness bar chart (10 % / 20 % / 30 % MNAR)

Requires:
    - Trained models in outputs/models/  (from train_all_models.py)
    - Robustness results in outputs/results/robustness_results.csv
      (from robustness_simulation.py)

Bugs fixed:
    B3  – joblib.load() returned a plain dict, not a DeepTriageCN instance.
           Fix: DeepTriageCN.load_from_file() factory method.

    B5  – fnr_data values were plain floats; plot_fnr_by_subgroup expected
           (fnr, n) tuples for confidence-interval computation.
           Fix: values are now (fnr_float, n_int) tuples.

    L4  – _scaled(X_val_raw) created a fresh StructuredEncoder and called
           fit_transform() on validation data → data leakage.
           Fix: the model's own fitted structured_encoder.transform() is used.

    L5  – comp_lens used str.len() (character count), but the paper defines
           "sparse" as ≤ 3 *words* (Section 4.7).
           Fix: str.split().str.len() for word count.

Usage:
    python scripts/generate_all_figures.py --config config.yaml
"""

import argparse
import yaml
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    get_raw_structured,
    preprocess_structured,
    preprocess_text,
    get_labels,
)
from src.fusion_model import DeepTriageCN
from src.tabnet_model import TabNetWrapper
from visualization.plot_roc import plot_roc_overall, plot_roc_geriatric
from visualization.plot_calibration import plot_calibration_curve
from visualization.plot_decision_curve import plot_decision_curve
from visualization.plot_robustness import plot_robustness_bars
from visualization.plot_shap import plot_shap_beeswarm
from visualization.plot_error_analysis import plot_error_distribution, plot_fnr_by_subgroup
from evaluation.calibration import brier_score
from evaluation.error_analysis import error_analysis_by_subgroup

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    fig_dir      = Path(config["output_paths"]["figures_dir"])
    models_dir   = Path(config["output_paths"]["models_dir"])
    results_dir  = Path(config["output_paths"]["results_dir"])
    threshold    = config["evaluation"]["optimal_threshold"]
    random_state = config["random_state"]
    fig_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    train_df, val_df = load_and_split_data(
        config["data"]["raw_data_path"],
        training_cutoff_date=config["data"]["training_cutoff_date"],
        validation_start_date=config["data"]["validation_start_date"],
        random_state=random_state,
    )

    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)
    _, X_val_std, _        = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val         = get_labels(train_df, val_df)

    # ------------------------------------------------------------------
    # 2. Load models  (Bug B3 fix: factory method for DeepTriageCN)
    # ------------------------------------------------------------------
    logger.info("Loading models …")
    deeptriage = DeepTriageCN.load_from_file(str(models_dir / "deeptriage_cn.pkl"))
    tabnet     = TabNetWrapper(); tabnet.load(str(models_dir / "tabnet"))
    vit_xgb    = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf         = joblib.load(models_dir / "random_forest.pkl")
    text_lr    = joblib.load(models_dir / "text_only_lr.pkl")
    val_scores = pd.read_csv(models_dir / "clinical_scores_val.csv")

    # BERT embeddings (frozen, identical across runs)
    text_enc = deeptriage.text_encoder
    val_emb  = text_enc.encode(val_texts.tolist())

    # ------------------------------------------------------------------
    # 3. Collect all model predictions
    # ------------------------------------------------------------------
    logger.info("Generating predictions …")
    preds = {
        "DeepTriage-CN":         deeptriage.predict_proba(
                                     val_texts.tolist(), X_val_raw),
        "TabNet":                tabnet.predict_proba(X_val_raw)[:, 1],
        "XGBoost (Vitals-only)": vit_xgb.predict_proba(X_val_std)[:, 1],
        "Random Forest":         rf.predict_proba(X_val_std)[:, 1],
        "Text-Only (BERT+LR)":  text_lr.predict_proba(val_emb)[:, 1],
        "NEWS2": val_scores["NEWS2"].values / 20.0,
        "MEWS":  val_scores["MEWS"].values  / 15.0,
        "ESI":  (6 - val_scores["ESI"].values) / 5.0,
    }

    # ------------------------------------------------------------------
    # 4. Figure 2a – Overall ROC
    # ------------------------------------------------------------------
    logger.info("Figure 2a: Overall ROC …")
    roc_data = {}
    for name, s in preds.items():
        fpr, tpr, _ = roc_curve(y_val, s)
        roc_data[name] = {
            "fpr":   fpr,
            "tpr":   tpr,
            "auroc": roc_auc_score(y_val, s),
        }
    plot_roc_overall(
        roc_data,
        save_path=str(fig_dir / "figure2a_roc_overall.tiff"),
    )

    # ------------------------------------------------------------------
    # 5. Figure 2b – Geriatric subgroup ROC
    # ------------------------------------------------------------------
    logger.info("Figure 2b: Geriatric ROC …")
    elder_mask = (val_df["age"] >= 65).values
    roc_ger = {}
    for name in ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)", "NEWS2", "ESI"]:
        fpr, tpr, _ = roc_curve(y_val[elder_mask], preds[name][elder_mask])
        roc_ger[name] = {
            "fpr":   fpr,
            "tpr":   tpr,
            "auroc": roc_auc_score(y_val[elder_mask], preds[name][elder_mask]),
        }
    plot_roc_geriatric(
        roc_ger,
        save_path=str(fig_dir / "figure2b_roc_geriatric.tiff"),
    )

    # ------------------------------------------------------------------
    # 6. Figure 3 – Calibration
    # ------------------------------------------------------------------
    logger.info("Figure 3: Calibration …")
    bs = brier_score(y_val, preds["DeepTriage-CN"])
    plot_calibration_curve(
        y_val, preds["DeepTriage-CN"],
        brier_score=bs,
        save_path=str(fig_dir / "figure3_calibration.tiff"),
    )

    # ------------------------------------------------------------------
    # 7. Figure 4 – Decision Curve Analysis
    # ------------------------------------------------------------------
    logger.info("Figure 4: Decision Curve …")
    plot_decision_curve(
        y_val,
        preds["DeepTriage-CN"],
        news2_scores=val_scores["NEWS2"].values,
        save_path=str(fig_dir / "figure4_decision_curve.tiff"),
    )

    # ------------------------------------------------------------------
    # 8. Figure 5 – SHAP beeswarm
    # Bug L4 fix: use the model's own fitted structured_encoder.transform()
    # instead of creating a fresh encoder fitted on validation data.
    # ------------------------------------------------------------------
    logger.info("Figure 5: SHAP beeswarm …")
    X_val_std_for_shap = deeptriage.structured_encoder.transform(X_val_raw)
    X_fused_val = np.hstack([val_emb, X_val_std_for_shap])
    feature_names = (
        [f"text_{i}" for i in range(768)]
        + ["age", "sex", "temperature", "heart_rate",
           "respiratory_rate", "sbp", "dbp", "spo2"]
    )
    plot_shap_beeswarm(
        deeptriage.classifier,
        X_fused_val,
        feature_names,
        save_path=str(fig_dir / "figure5_shap.tiff"),
    )

    # ------------------------------------------------------------------
    # 9. Figure 6a – Error distribution
    # ------------------------------------------------------------------
    logger.info("Figure 6a: Error distribution …")
    y_pred_bin = (preds["DeepTriage-CN"] >= threshold).astype(int)
    plot_error_distribution(
        y_val, y_pred_bin,
        save_path=str(fig_dir / "figure6a_error_distribution.tiff"),
    )

    # ------------------------------------------------------------------
    # 10. Figure 6b – FNR by subgroup
    # Bug L5 fix: count words (str.split().str.len()), not characters.
    # Bug B5 fix: fnr_data values must be (fnr_float, n_int) tuples.
    # ------------------------------------------------------------------
    logger.info("Figure 6b: FNR by subgroup …")
    comp_lens = (
        val_df["chief_complaint"]
        .fillna("")
        .str.split()
        .str.len()
        .values
    )  # word count (Section 4.7: sparse ≤ 3 words)

    err_res = error_analysis_by_subgroup(
        y_val, y_pred_bin,
        ages=val_df["age"].values,
        complaint_lengths=comp_lens,
        age_threshold=65,
        sparse_narrative_threshold=3,    # paper definition: ≤ 3 words
    )

    # Build (fnr, n) tuples expected by plot_fnr_by_subgroup
    fnr_data = {
        "young_sparse": (err_res["young_sparse"]["fnr"],
                         err_res["young_sparse"]["n"]),
        "elder_sparse": (err_res["elder_sparse"]["fnr"],
                         err_res["elder_sparse"]["n"]),
        "young_rich":   (err_res["young_rich"]["fnr"],
                         err_res["young_rich"]["n"]),
        "elder_rich":   (err_res["elder_rich"]["fnr"],
                         err_res["elder_rich"]["n"]),
    }
    plot_fnr_by_subgroup(
        fnr_data,
        save_path=str(fig_dir / "figure6b_fnr_subgroup.tiff"),
    )

    # ------------------------------------------------------------------
    # 11. Supplementary Figure 1 – Robustness bars (all 3 missing levels)
    # ------------------------------------------------------------------
    logger.info("Supplementary Figure 1: Robustness bars …")
    rob_df = pd.read_csv(results_dir / "robustness_results.csv")

    # Use 30 % level for the main supplementary bar chart
    rob_30 = rob_df[rob_df["Missing_Proportion"] == 0.30]
    base_dict = dict(zip(rob_30["Model"], rob_30["Baseline_AUROC"]))
    deg_dict  = dict(zip(rob_30["Model"], rob_30["Degraded_AUROC"]))

    plot_robustness_bars(
        base_dict, deg_dict,
        save_path=str(fig_dir / "supp_figure1_robustness.tiff"),
    )

    logger.info("=" * 60)
    logger.info("All figures saved → %s", fig_dir)


if __name__ == "__main__":
    main()
