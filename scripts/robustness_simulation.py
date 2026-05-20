#!/usr/bin/env python3
"""
robustness_simulation.py

Runs the data-degradation robustness simulation (Section 3.7) and computes
performance retention for DeepTriage-CN, TabNet, and Vitals-only XGBoost.

The simulation introduces compounded degradation at three levels
(10 %, 20 %, 30 % MNAR missingness + Gaussian noise) and reports:
    - Baseline AUROC (full clean validation data)
    - Degraded AUROC at each missing proportion
    - Retention (%) = degraded / baseline × 100

Outputs:
    outputs/results/robustness_results.csv   — full table (3 models × 3 levels)
    outputs/results/robustness_summary.txt   — human-readable summary

Bugs fixed:
    B3  – joblib.load() returned a plain dict, not a DeepTriageCN instance.
           Fix: DeepTriageCN.load_from_file() factory method used.

    L1  – preprocess_structured() was called returning already-standardised
           arrays, which were then passed through scaler.transform() again →
           double standardisation of val data.
           Fix: get_raw_structured() is used for DeepTriageCN / TabNet;
           preprocess_structured() only for VitalsXGB.

    L3  – Only the 30 % level was tested.
           Fix: all three levels (10 %, 20 %, 30 %) are now iterated.

Usage:
    python scripts/robustness_simulation.py --config config.yaml
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
from src.robustness import simulate_degraded_data
from src.structured_encoder import STRUCTURED_FEATURE_NAMES, StructuredEncoder
from evaluation.metrics import compute_all_binary_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    rob_cfg      = config["robustness"]
    threshold    = config["evaluation"]["optimal_threshold"]
    output_dir   = Path(config["output_paths"]["results_dir"])
    models_dir   = Path(config["output_paths"]["models_dir"])
    random_state = config["random_state"]
    output_dir.mkdir(parents=True, exist_ok=True)

    missing_proportions = rob_cfg.get("missing_proportions", [0.10, 0.20, 0.30])
    noise_sigma         = rob_cfg.get("noise_sigma_multiplier", 0.5)

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

    # Raw arrays (for DeepTriageCN and TabNet)
    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)

    # Standardised arrays (for VitalsXGB — it does not standardise internally)
    X_train_std, X_val_std, baseline_encoder = preprocess_structured(
        train_df, val_df
    )

    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    # ------------------------------------------------------------------
    # 2. Load trained models
    # ------------------------------------------------------------------
    logger.info("Loading models …")

    # DeepTriage-CN (Bug B3 fix: use factory method)
    deeptriage = DeepTriageCN.load_from_file(
        str(models_dir / "deeptriage_cn.pkl")
    )

    # TabNet
    tabnet = TabNetWrapper()
    tabnet.load(str(models_dir / "tabnet"))

    # VitalsXGB
    vit_xgb = joblib.load(models_dir / "vitals_only_xgb.pkl")

    # Pre-compute BERT embeddings once (text is not degraded)
    logger.info("Pre-computing BERT embeddings for validation set …")
    val_emb = deeptriage.text_encoder.encode(val_texts.tolist())

    # ------------------------------------------------------------------
    # 3. Baseline AUROCs (clean data)
    # ------------------------------------------------------------------
    logger.info("Computing baseline AUROCs (clean data) …")

    # DeepTriage-CN: use raw val data (model standardises internally)
    dt_proba_clean = deeptriage.predict_proba(val_texts.tolist(), X_val_raw)

    # TabNet: raw val data (BN internally)
    tn_proba_clean = tabnet.predict_proba(X_val_raw)[:, 1]

    # VitalsXGB: standardised val data
    xgb_proba_clean = vit_xgb.predict_proba(X_val_std)[:, 1]

    baseline = {
        "DeepTriage-CN":         _auroc(y_val, dt_proba_clean,  threshold),
        "TabNet":                _auroc(y_val, tn_proba_clean,  threshold),
        "XGBoost (Vitals-only)": _auroc(y_val, xgb_proba_clean, threshold),
    }
    logger.info("Baseline AUROCs: %s", baseline)

    # ------------------------------------------------------------------
    # 4. Degraded AUROCs across all missing proportions (Bug L3 fix)
    # ------------------------------------------------------------------
    # simulate_degraded_data returns a dict: {proportion: degraded_X_raw}
    all_degraded_raw = simulate_degraded_data(
        X_val_raw,
        STRUCTURED_FEATURE_NAMES,
        missing_proportions=missing_proportions,
        noise_sigma=noise_sigma,
        seed=random_state,
    )

    # For VitalsXGB, the degraded raw array must be standardised using the
    # training-set encoder (not re-fitted on degraded data → no leakage).
    rows = []
    for prop in missing_proportions:
        X_deg_raw = all_degraded_raw[prop]          # degraded raw

        # DeepTriage-CN: pass raw degraded data (model uses its fitted encoder)
        dt_proba_deg = deeptriage.predict_proba(val_texts.tolist(), X_deg_raw)

        # TabNet: raw degraded data
        tn_proba_deg = tabnet.predict_proba(X_deg_raw)[:, 1]

        # VitalsXGB: standardise degraded raw with training encoder
        X_deg_std = baseline_encoder.transform(X_deg_raw)
        xgb_proba_deg = vit_xgb.predict_proba(X_deg_std)[:, 1]

        degraded = {
            "DeepTriage-CN":         _auroc(y_val, dt_proba_deg,  threshold),
            "TabNet":                _auroc(y_val, tn_proba_deg,  threshold),
            "XGBoost (Vitals-only)": _auroc(y_val, xgb_proba_deg, threshold),
        }

        for model_name in baseline:
            rows.append({
                "Model":             model_name,
                "Missing_Proportion": prop,
                "Baseline_AUROC":    round(baseline[model_name],  4),
                "Degraded_AUROC":    round(degraded[model_name],  4),
                "Retention_pct":     round(
                    degraded[model_name] / baseline[model_name] * 100, 2
                ),
            })

        logger.info(
            "[%.0f%% MNAR + noise] DeepTriage-CN %.4f (%.1f%%), "
            "TabNet %.4f (%.1f%%), XGBoost %.4f (%.1f%%)",
            prop * 100,
            degraded["DeepTriage-CN"],
            degraded["DeepTriage-CN"] / baseline["DeepTriage-CN"] * 100,
            degraded["TabNet"],
            degraded["TabNet"] / baseline["TabNet"] * 100,
            degraded["XGBoost (Vitals-only)"],
            degraded["XGBoost (Vitals-only)"] / baseline["XGBoost (Vitals-only)"] * 100,
        )

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    results_df = pd.DataFrame(rows)
    results_df.to_csv(output_dir / "robustness_results.csv", index=False)
    logger.info("Full results → %s", output_dir / "robustness_results.csv")

    # Human-readable summary (30 % level, matching main paper text)
    summary_rows_30 = results_df[
        results_df["Missing_Proportion"] == 0.30
    ].copy()
    summary_path = output_dir / "robustness_summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Robustness Simulation — 30 % MNAR + Gaussian Noise\n")
        f.write("=" * 60 + "\n")
        f.write(summary_rows_30.to_string(index=False))
        f.write("\n")
    logger.info("Summary (30%%) → %s", summary_path)

    print("\n=== Robustness Results (all levels) ===")
    print(results_df.to_string(index=False))
    logger.info("Robustness simulation complete.")


def _auroc(y_true, proba, threshold):
    """Thin wrapper returning only the AUROC from compute_all_binary_metrics."""
    return compute_all_binary_metrics(y_true, proba, threshold)["auroc"]


if __name__ == "__main__":
    main()
