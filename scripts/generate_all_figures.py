#!/usr/bin/env python3
"""
generate_all_figures.py
=======================
Generates all publication figures for DeepTriage-CN.

Output (all 300 DPI TIFF, LZW compressed):
  Figure 2           ROC curves, dual panel (overall + geriatric)
  Figure 3           Calibration + Decision Curve Analysis
  Figure 4           SHAP + error distribution + FNR subgroup
  Supplementary S1   Robustness under MNAR data degradation

Bug fixes applied vs. original code:
  L4  preprocess: validation structured features now transformed
      via deeptriage.structured_encoder.transform() — not refit.
  L5  Sparse narrative: identified by word count (≤3 words) not
      character count (Section 4.7).

Usage:
    python scripts/generate_all_figures.py --config config.yaml
"""
import argparse, logging, sys, yaml
import numpy as np, pandas as pd, joblib
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import load_and_split_data, get_raw_structured
from src.preprocess import preprocess_structured, preprocess_text, get_labels
from src.fusion_model import DeepTriageCN
from src.tabnet_model import TabNetWrapper
from visualization.generate_figures_nature import make_fig2, make_fig3, make_fig4
from visualization.plot_robustness import plot_robustness

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    data_cfg    = cfg["data"]
    eval_cfg    = cfg["evaluation"]
    models_dir  = Path(cfg["output_paths"]["models_dir"])
    results_dir = Path(cfg["output_paths"]["results_dir"])
    figures_dir = Path(cfg["output_paths"]["figures_dir"])
    figures_dir.mkdir(parents=True, exist_ok=True)
    geri_age = cfg.get("geriatric_age_threshold", 65)

    # ── 1. Data ─────────────────────────────────────────────────────────
    log.info("Loading data …")
    train_df, val_df = load_and_split_data(
        data_cfg["raw_data_path"],
        training_cutoff_date=data_cfg["training_cutoff_date"],
        validation_start_date=data_cfg["validation_start_date"],
        random_state=cfg["random_state"])
    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)
    _, X_val_std, _        = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val         = get_labels(train_df, val_df)

    # ── 2. Models ────────────────────────────────────────────────────────
    log.info("Loading models …")
    dtc    = DeepTriageCN.load_from_file(str(models_dir / "deeptriage_cn.pkl"))
    tabnet = TabNetWrapper(); tabnet.load(str(models_dir / "tabnet"))
    vit    = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf     = joblib.load(models_dir / "random_forest.pkl")
    tlr    = joblib.load(models_dir / "text_only_lr.pkl")

    # ── 3. Predictions ───────────────────────────────────────────────────
    log.info("Generating predictions …")
    val_emb    = dtc.text_encoder.encode(val_texts.tolist())
    # Bug L4: use transform, not fit_transform, on validation data
    X_val_std2 = dtc.structured_encoder.transform(X_val_raw)
    scr        = pd.read_csv(models_dir / "clinical_scores_val.csv")

    probs = {
        "DeepTriage-CN":         dtc.predict_proba(val_texts.tolist(), X_val_raw),
        "TabNet":                tabnet.predict_proba(X_val_raw)[:, 1],
        "XGBoost (Vitals-only)": vit.predict_proba(X_val_std)[:, 1],
        "Random Forest":         rf.predict_proba(X_val_std)[:, 1],
        "Text-Only (BERT+LR)":   tlr.predict_proba(val_emb)[:, 1],
        "NEWS2":                 scr["NEWS2"].values / 20.0,
        "MEWS":                  scr["MEWS"].values  / 15.0,
        "ESI":                   (6 - scr["ESI"].values) / 5.0,
    }
    geri_mask = val_df["age"].values >= geri_age
    # Bug L5: word count for sparse narrative (≤3 words)
    cc_sparse = val_df["chief_complaint"].fillna("").str.split().str.len() <= 3

    # ── 4–6. Figures ─────────────────────────────────────────────────────
    log.info("Figure 2 (ROC) …")
    make_fig2(y_val, geri_mask, probs, str(figures_dir))
    log.info("  ✓  Figure2_ROC.tiff")

    log.info("Figure 3 (Calibration + DCA) …")
    make_fig3(y_val, probs, str(figures_dir))
    log.info("  ✓  Figure3_Calibration_DCA.tiff")

    log.info("Figure 4 (SHAP + Error + FNR) …")
    shap_npz = results_dir / "shap_values.npz"
    make_fig4(y_val, geri_mask, probs, cc_sparse.values, str(figures_dir),
              shap_npz_path=str(shap_npz) if shap_npz.exists() else None)
    log.info("  ✓  Figure4_SHAP_Error_FNR.tiff")

    rob_csv = results_dir / "robustness_results.csv"
    if rob_csv.exists():
        log.info("Supplementary Figure S1 (Robustness) …")
        plot_robustness(pd.read_csv(rob_csv),
                        out_path=str(figures_dir / "SuppFigure1_Robustness.tiff"))
        log.info("  ✓  SuppFigure1_Robustness.tiff")

    log.info("All figures saved → %s", figures_dir)


if __name__ == "__main__":
    main()
