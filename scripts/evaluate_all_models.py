#!/usr/bin/env python3
"""
evaluate_all_models.py

Evaluates all trained models on the temporal validation set and reproduces
the core quantitative results from the paper:

    Table 2 : AUROC, AUPRC, sensitivity, specificity, F1, accuracy
    Bootstrap 95 % CIs (n = 1 000 iterations)
    DeLong tests (DeepTriage-CN vs each comparator)
    Continuous NRI (DeepTriage-CN vs NEWS2 and ESI)
    Calibration metrics (Brier score, Hosmer–Lemeshow, slope/intercept)

Outputs are written to outputs/results/.

Bug fix applied:
    B3 – The original script did:
           deeptriage = joblib.load(...)
         which returned the raw dict stored by save(), not a DeepTriageCN
         instance, so .predict_proba() raised AttributeError.

         Fix: use DeepTriageCN.load_from_file() — a class-level factory that
         reconstructs a fully initialised object from the saved dict.

Usage:
    python scripts/evaluate_all_models.py --config config.yaml
"""

import argparse
import yaml
import sys
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

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
from evaluation.metrics import compute_all_binary_metrics
from evaluation.calibration import compute_calibration_metrics
from evaluation.bootstrap import bootstrap_confidence_intervals
from evaluation.statistical_tests import delong_test, continuous_nri

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Evaluate all DeepTriage-CN models.")
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    threshold    = config["evaluation"]["optimal_threshold"]
    n_bootstrap  = config["evaluation"]["n_bootstrap"]
    output_dir   = Path(config["output_paths"]["results_dir"])
    models_dir   = Path(config["output_paths"]["models_dir"])
    random_state = config["random_state"]
    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Load and preprocess data
    # ------------------------------------------------------------------
    logger.info("Loading data …")
    train_df, val_df = load_and_split_data(
        config["data"]["raw_data_path"],
        training_cutoff_date=config["data"]["training_cutoff_date"],
        validation_start_date=config["data"]["validation_start_date"],
        random_state=random_state,
    )

    # Raw arrays — for DeepTriageCN and TabNet
    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)

    # Standardised arrays — for VitalsXGB, RF, TextLR
    _, X_val_std, _ = preprocess_structured(train_df, val_df)

    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    # ------------------------------------------------------------------
    # 2. Load trained models
    # ------------------------------------------------------------------
    logger.info("Loading models …")

    # DeepTriage-CN — use class-level factory (Bug B3 fix)
    deeptriage = DeepTriageCN.load_from_file(
        str(models_dir / "deeptriage_cn.pkl")
    )

    # TabNet
    tabnet = TabNetWrapper()
    tabnet.load(str(models_dir / "tabnet"))

    # Sklearn-style baselines
    vit_xgb = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf       = joblib.load(models_dir / "random_forest.pkl")
    text_lr  = joblib.load(models_dir / "text_only_lr.pkl")

    # Pre-computed clinical scores (written by train_all_models.py)
    val_scores = pd.read_csv(models_dir / "clinical_scores_val.csv")

    # ------------------------------------------------------------------
    # 3. Generate predictions
    # ------------------------------------------------------------------
    logger.info("Generating predictions …")

    # BERT embeddings for text-only LR
    text_enc = deeptriage.text_encoder
    val_emb  = text_enc.encode(val_texts.tolist())

    preds = {
        # Multimodal — raw data; DeepTriageCN standardises internally
        "DeepTriage-CN":         deeptriage.predict_proba(
                                     val_texts.tolist(), X_val_raw
                                 ),
        # TabNet — raw data (BN internally)
        "TabNet":                tabnet.predict_proba(X_val_raw)[:, 1],
        # Baselines — standardised data
        "XGBoost (Vitals-only)": vit_xgb.predict_proba(X_val_std)[:, 1],
        "Random Forest":         rf.predict_proba(X_val_std)[:, 1],
        "Text-Only (BERT+LR)":  text_lr.predict_proba(val_emb)[:, 1],
        # Clinical scores — normalised to [0, 1] probability-like scale
        "NEWS2": val_scores["NEWS2"].values / 20.0,
        "MEWS":  val_scores["MEWS"].values  / 15.0,
        "ESI":  (6 - val_scores["ESI"].values) / 5.0,   # invert (1 = urgent)
    }

    # ------------------------------------------------------------------
    # 4. Table 2 — binary classification metrics
    # ------------------------------------------------------------------
    logger.info("Computing classification metrics (Table 2) …")
    results = []
    for name, scores in preds.items():
        metrics = compute_all_binary_metrics(y_val, scores, threshold)
        metrics["Model"] = name
        results.append(metrics)

    results_df = pd.DataFrame(results).set_index("Model")
    results_df.round(4).to_csv(output_dir / "table2_metrics.csv")
    logger.info("Table 2 saved → %s", output_dir / "table2_metrics.csv")
    print("\n=== Table 2 ===")
    print(results_df.round(4).to_string())

    # ------------------------------------------------------------------
    # 5. Bootstrap confidence intervals
    # ------------------------------------------------------------------
    logger.info("Computing bootstrap CIs (n = %d) …", n_bootstrap)
    ci_records = {}
    for name in ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)", "NEWS2", "ESI"]:
        ci = bootstrap_confidence_intervals(
            y_val, preds[name],
            n_bootstrap=n_bootstrap,
            random_seed=random_state,
        )
        ci_records[name] = ci

    pd.DataFrame(ci_records).T.to_csv(output_dir / "bootstrap_cis.csv")
    logger.info("Bootstrap CIs saved → %s", output_dir / "bootstrap_cis.csv")

    # ------------------------------------------------------------------
    # 6. DeLong tests vs DeepTriage-CN
    # ------------------------------------------------------------------
    logger.info("Running DeLong tests …")
    comparators = ["TabNet", "XGBoost (Vitals-only)", "Random Forest",
                   "Text-Only (BERT+LR)", "NEWS2", "MEWS", "ESI"]
    delong_rows = []
    for name in comparators:
        z, p = delong_test(y_val, preds["DeepTriage-CN"], preds[name])
        delong_rows.append({"Comparison": f"DeepTriage-CN vs {name}",
                            "z": round(z, 4), "p_value": round(p, 4)})
    delong_df = pd.DataFrame(delong_rows)
    delong_df.to_csv(output_dir / "delong_tests.csv", index=False)
    print("\n=== DeLong Tests ===")
    print(delong_df.to_string(index=False))
    logger.info("DeLong tests saved → %s", output_dir / "delong_tests.csv")

    # ------------------------------------------------------------------
    # 7. Continuous NRI
    # ------------------------------------------------------------------
    logger.info("Computing continuous NRI …")
    nri_rows = []
    for ref_name in ["NEWS2", "ESI"]:
        nri_total, nri_events, nri_nonevents = continuous_nri(
            y_val, preds[ref_name], preds["DeepTriage-CN"]
        )
        nri_rows.append({
            "Comparison":    f"vs {ref_name}",
            "NRI_total":     round(nri_total, 4),
            "NRI_events":    round(nri_events, 4),
            "NRI_nonevents": round(nri_nonevents, 4),
        })
    nri_df = pd.DataFrame(nri_rows)
    nri_df.to_csv(output_dir / "nri_results.csv", index=False)
    print("\n=== Continuous NRI ===")
    print(nri_df.to_string(index=False))

    # ------------------------------------------------------------------
    # 8. Calibration metrics for DeepTriage-CN
    # ------------------------------------------------------------------
    logger.info("Computing calibration metrics …")
    cal = compute_calibration_metrics(y_val, preds["DeepTriage-CN"])
    cal_series = pd.Series(cal)
    cal_series.to_csv(output_dir / "calibration_metrics.csv", header=["value"])
    print("\n=== Calibration ===")
    print(cal_series)

    logger.info("=" * 60)
    logger.info("Evaluation complete.  All outputs in: %s", output_dir)


if __name__ == "__main__":
    main()
