#!/usr/bin/env python3
"""
scripts/evaluate_all_models.py
===============================
Runs the complete evaluation pipeline for all models and exports
Tables 2, 3, and 4 of the paper as CSV files.

Usage:
    python scripts/evaluate_all_models.py --config config.yaml
"""
import argparse, logging, sys, json, yaml, joblib
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data, get_raw_structured,
    preprocess_structured, preprocess_text, get_labels,
)
from src.fusion_model import DeepTriageCN
from src.tabnet_model import TabNetWrapper
from evaluation.metrics import (
    bootstrap_auroc, delong_test, youden_metrics,
    calibration_metrics, hosmer_lemeshow,
    nri, alert_burden, interaction_contrast,
)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

MODEL_ORDER = [
    "DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)",
    "Random Forest", "Text-Only (BERT+LR)",
    "NEWS2", "MEWS", "ESI",
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    data_cfg    = cfg["data"]
    eval_cfg    = cfg["evaluation"]
    models_dir  = Path(cfg["output_paths"]["models_dir"])
    results_dir = Path(cfg["output_paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    youden_thr  = eval_cfg["optimal_threshold"]
    n_boot      = eval_cfg["n_bootstrap"]
    geri_age    = cfg.get("geriatric_age_threshold", 65)
    seed        = cfg["random_state"]

    # ── 1. Load data ────────────────────────────────────────────────────
    log.info("Loading data …")
    train_df, val_df = load_and_split_data(
        data_cfg["raw_data_path"],
        training_cutoff_date=data_cfg["training_cutoff_date"],
        validation_start_date=data_cfg["validation_start_date"],
        random_state=seed)
    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)
    _, X_val_std, _        = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val         = get_labels(train_df, val_df)

    geri_mask = val_df["age"].values >= geri_age
    log.info("Val N=%d  admitted=%.3f  geriatric=%d",
             len(y_val), y_val.mean(), geri_mask.sum())

    # ── 2. Load models & generate probabilities ──────────────────────────
    log.info("Loading models and generating predictions …")
    dtc    = DeepTriageCN.load_from_file(str(models_dir / "deeptriage_cn.pkl"))
    tabnet = TabNetWrapper(); tabnet.load(str(models_dir / "tabnet"))
    vit    = joblib.load(models_dir / "vitals_only_xgb.pkl")
    rf     = joblib.load(models_dir / "random_forest.pkl")
    tlr    = joblib.load(models_dir / "text_only_lr.pkl")
    val_emb = dtc.text_encoder.encode(val_texts.tolist())

    scr = pd.read_csv(models_dir / "clinical_scores_val.csv")

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

    # ── 3. TABLE 2: overall performance ─────────────────────────────────
    log.info("\nComputing Table 2 (overall performance) …")
    primary = "DeepTriage-CN"
    t2_rows = []
    for name in MODEL_ORDER:
        prob = probs[name]
        a  = bootstrap_auroc(y_val, prob, n_bootstrap=n_boot, seed=seed)
        ym = youden_metrics(y_val, prob, threshold=youden_thr,
                            n_bootstrap=n_boot, seed=seed)
        d  = delong_test(y_val, probs[primary], prob, n_bootstrap=n_boot, seed=seed)
        t2_rows.append({
            "model":            name,
            "auroc":            a["auroc"],
            "auroc_ci_lo":      a["ci_lo"],
            "auroc_ci_hi":      a["ci_hi"],
            "auprc":            a["auprc"],
            "auprc_ci_lo":      a["ci_auprc_lo"],
            "auprc_ci_hi":      a["ci_auprc_hi"],
            "sensitivity":      ym["sensitivity"],
            "sens_ci_lo":       ym["sens_ci"][0],
            "sens_ci_hi":       ym["sens_ci"][1],
            "specificity":      ym["specificity"],
            "spec_ci_lo":       ym["spec_ci"][0],
            "spec_ci_hi":       ym["spec_ci"][1],
            "accuracy":         ym["accuracy"],
            "acc_ci_lo":        ym["acc_ci"][0],
            "acc_ci_hi":        ym["acc_ci"][1],
            "ppv":              ym["ppv"],
            "delta_auroc_vs_primary": d["delta_auroc"],
            "delta_ci_lo":      d["ci_lo"],
            "delta_ci_hi":      d["ci_hi"],
            "delong_p":         d["p_value"],
        })
    t2 = pd.DataFrame(t2_rows)
    t2.to_csv(results_dir / "table2_performance.csv", index=False)
    log.info("  Saved → %s", results_dir / "table2_performance.csv")
    log.info(t2[["model","auroc","sensitivity","specificity","ppv","delong_p"]].to_string(index=False))

    # ── 4. NRI vs NEWS2 and ESI ─────────────────────────────────────────
    log.info("\nComputing NRI …")
    nri_rows = []
    for ref in ["NEWS2", "ESI"]:
        r = nri(y_val, probs[primary], probs[ref],
                n_bootstrap=n_boot, seed=seed)
        nri_rows.append({"reference": ref, **r})
        log.info("  NRI vs %s: %.3f (%.3f–%.3f)  p=%.4f",
                 ref, r["nri"], r["ci_lo"], r["ci_hi"], r["p_value"])
    pd.DataFrame(nri_rows).to_csv(results_dir / "nri_results.csv", index=False)

    # ── 5. TABLE 3: geriatric subgroup calibration ───────────────────────
    log.info("\nComputing Table 3 (geriatric subgroup) …")
    y_g   = y_val[geri_mask]
    geri_models = ["DeepTriage-CN","TabNet","XGBoost (Vitals-only)","NEWS2","ESI"]
    t3_rows = []
    for name in geri_models:
        p_g = probs[name][geri_mask]
        a   = bootstrap_auroc(y_g, p_g, n_bootstrap=n_boot, seed=seed)
        cm  = calibration_metrics(y_g, p_g, n_bootstrap=n_boot, seed=seed)
        hl  = hosmer_lemeshow(y_g, p_g)
        t3_rows.append({
            "model":         name,
            "auroc":         a["auroc"],
            "auroc_ci_lo":   a["ci_lo"],
            "auroc_ci_hi":   a["ci_hi"],
            "brier":         cm["brier"],
            "brier_ci_lo":   cm["brier_ci"][0],
            "brier_ci_hi":   cm["brier_ci"][1],
            "cal_slope":     cm["cal_slope"],
            "slope_ci_lo":   cm["slope_ci"][0],
            "slope_ci_hi":   cm["slope_ci"][1],
            "cal_intercept": cm["cal_intercept"],
            "int_ci_lo":     cm["intercept_ci"][0],
            "int_ci_hi":     cm["intercept_ci"][1],
            "hl_p_value":    hl["p_value"],
        })
    t3 = pd.DataFrame(t3_rows)
    t3.to_csv(results_dir / "table3_geriatric_calibration.csv", index=False)
    log.info("  Saved → %s", results_dir / "table3_geriatric_calibration.csv")
    log.info(t3[["model","auroc","brier","cal_slope","hl_p_value"]].to_string(index=False))

    # ── 6. Interaction contrasts ─────────────────────────────────────────
    log.info("\nComputing interaction contrasts (age × model) …")
    ic_rows = []
    for ref in ["NEWS2", "ESI", "TabNet", "XGBoost (Vitals-only)"]:
        ic = interaction_contrast(
            y_val, probs[primary], probs[ref], geri_mask,
            n_bootstrap=n_boot, seed=seed)
        ic_rows.append({"comparison": f"{primary} vs {ref}", **ic})
        log.info("  %s vs %s: Δ=%.3f (%.3f–%.3f) p=%.4f",
                 primary, ref, ic["delta_diff"],
                 ic["ci_lo"], ic["ci_hi"], ic["p_value"])
    pd.DataFrame(ic_rows).to_csv(
        results_dir / "interaction_contrasts.csv", index=False)

    # ── 7. TABLE 4: alert burden ─────────────────────────────────────────
    log.info("\nComputing Table 4 (alert burden) …")
    t4 = alert_burden(y_val, probs[primary],
                      thresholds=cfg["alert_thresholds"])
    t4.to_csv(results_dir / "table4_alert_burden.csv", index=False)
    log.info("  Saved → %s", results_dir / "table4_alert_burden.csv")
    log.info(t4.to_string(index=False))

    # ── 8. Error analysis ─────────────────────────────────────────────────
    log.info("\nComputing error analysis (Section 4.7) …")
    pred = (probs[primary] >= youden_thr).astype(int)
    cc_words = val_df["chief_complaint"].fillna("").str.split().str.len()
    cc_sparse = cc_words <= 3

    error_rows = []
    for lab, mask in [
        ("Young (<65) + sparse CC",  (~geri_mask) & cc_sparse.values),
        ("Young (<65) + rich CC",    (~geri_mask) & ~cc_sparse.values),
        ("Elder (≥65) + sparse CC",  geri_mask   & cc_sparse.values),
        ("Elder (≥65) + rich CC",    geri_mask   & ~cc_sparse.values),
    ]:
        sy = y_val[mask]; sp = pred[mask]
        pos = sy.sum(); neg = (~sy.astype(bool)).sum()
        fn  = ((sp==0)&(sy==1)).sum()
        fp  = ((sp==1)&(sy==0)).sum()
        fnr = fn / max(pos, 1)
        fpr = fp / max(neg, 1)
        error_rows.append({
            "group": lab, "n": mask.sum(), "n_positive": pos,
            "fn": fn, "fp": fp, "fnr": fnr, "fpr_group": fpr,
        })
    ea = pd.DataFrame(error_rows)
    ea.to_csv(results_dir / "error_analysis.csv", index=False)
    log.info(ea[["group","n","fnr"]].to_string(index=False))

    # ── 9. Summary JSON ───────────────────────────────────────────────────
    summary = {
        "primary_model":   primary,
        "n_val":           int(len(y_val)),
        "youden_threshold": youden_thr,
        "geriatric_n":     int(geri_mask.sum()),
        "primary_auroc":   float(probs["DeepTriage-CN"] and
                           t2[t2.model==primary]["auroc"].iloc[0]),
        "primary_ppv_youden": float(
            t2[t2.model==primary]["ppv"].iloc[0]),
    }
    with open(results_dir / "evaluation_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log.info("\n" + "="*60)
    log.info("Evaluation complete.  Results in: %s", results_dir)
    log.info("="*60)


if __name__ == "__main__":
    main()
