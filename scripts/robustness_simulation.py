#!/usr/bin/env python3
"""
scripts/robustness_simulation.py
=================================
Simulates MNAR (missing-not-at-random) data degradation and evaluates
AUROC retention for DeepTriage-CN, TabNet, and XGBoost (Vitals-only).

Method (Section 3.7):
  Missingness probability: P(M|x) = inverse logistic decay centred on
  normal physiological range. Values near healthy baselines are
  preferentially removed, simulating low-acuity omission heuristics.
  Simultaneously, Gaussian noise ε ~ N(0, 0.5·σ_x) is injected into
  remaining continuous vital-sign values.

  Aggregate missingness proportions tested: 10%, 20%, 30%.
  Each condition repeated 5 times (n_repetitions in config.yaml).

  Reported metric: AUROC retention = AUROC_degraded / AUROC_baseline.

Outputs:
  outputs/results/robustness_results.csv

Usage:
    python scripts/robustness_simulation.py --config config.yaml
"""
import argparse, logging, sys, yaml, joblib
import numpy as np, pandas as pd
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data, get_raw_structured, preprocess_structured,
    preprocess_text, get_labels,
)
from src.fusion_model import DeepTriageCN
from src.tabnet_model import TabNetWrapper
from sklearn.metrics import roc_auc_score

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

# Normal physiological midpoints (used to centre decay function)
PHYS_MID = {
    0: 37.0,   # temperature  (index 2 in struct vector)
    1: 80.0,   # heart_rate
    2: 16.0,   # respiratory_rate
    3: 120.0,  # sbp
    4: 80.0,   # dbp
    5: 97.0,   # spo2
}
VITAL_COLS = [2, 3, 4, 5, 6, 7]   # indices in 8-dim struct vector (skip age, sex)


def _mnar_mask(X: np.ndarray, proportion: float,
               rng: np.random.Generator) -> np.ndarray:
    """
    Returns a boolean mask (True = mask out) for MNAR missingness.
    Probability of masking is highest for values near the physiological
    normal range (inverse logistic decay).
    """
    mask = np.zeros(X.shape, dtype=bool)
    n_vital = len(VITAL_COLS)
    # Total number of vital values to remove
    n_remove = int(round(proportion * X.shape[0] * n_vital))

    # Compute P(miss) for each vital × patient cell
    P = np.zeros((X.shape[0], n_vital))
    for j, col_idx in enumerate(VITAL_COLS):
        mid = PHYS_MID.get(j, 0.0)
        x_col = X[:, col_idx]
        sd = np.nanstd(x_col); sd = sd if sd > 0 else 1.0
        dist = np.abs(x_col - mid) / sd
        # Inverse logistic: closer to normal → higher P(miss)
        P[:, j] = 1.0 / (1.0 + np.exp(dist - 1.5))

    P_flat = P.flatten()
    P_flat /= P_flat.sum()
    chosen = rng.choice(len(P_flat), size=min(n_remove, len(P_flat)),
                        replace=False, p=P_flat)
    rows, cols_local = np.unravel_index(chosen, P.shape)
    for r, cl in zip(rows, cols_local):
        mask[r, VITAL_COLS[cl]] = True
    return mask


def _inject_noise(X: np.ndarray, sigma_mult: float = 0.5,
                  rng: np.random.Generator = None) -> np.ndarray:
    """Inject Gaussian noise ε ~ N(0, sigma_mult · σ_x) into vital sign columns."""
    X_noisy = X.copy()
    for col_idx in VITAL_COLS:
        col = X_noisy[:, col_idx]
        sd  = np.nanstd(col); sd = sd if sd > 0 else 1.0
        X_noisy[:, col_idx] += rng.normal(0, sigma_mult * sd, X_noisy.shape[0])
    return X_noisy


def _apply_degradation(X_raw: np.ndarray, enc, proportion: float,
                       sigma_mult: float, rng: np.random.Generator) -> np.ndarray:
    """Apply MNAR mask + noise and return standardised feature matrix."""
    X_deg = X_raw.copy().astype(float)
    mask  = _mnar_mask(X_deg, proportion, rng)
    X_deg[mask] = np.nan
    X_deg = _inject_noise(X_deg, sigma_mult=sigma_mult, rng=rng)
    return enc.transform(X_deg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    args = ap.parse_args()
    with open(args.config) as f: cfg = yaml.safe_load(f)

    data_cfg   = cfg["data"]
    rob_cfg    = cfg["robustness"]
    models_dir = Path(cfg["output_paths"]["models_dir"])
    results_dir= Path(cfg["output_paths"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    proportions   = rob_cfg["missing_proportions"]
    sigma_mult    = rob_cfg["noise_sigma_multiplier"]
    n_reps        = rob_cfg["n_repetitions"]

    # ── Load data ────────────────────────────────────────────────────────
    log.info("Loading data …")
    _, val_df = load_and_split_data(
        data_cfg["raw_data_path"],
        training_cutoff_date=data_cfg["training_cutoff_date"],
        validation_start_date=data_cfg["validation_start_date"],
        random_state=cfg["random_state"])
    _, X_val_raw      = get_raw_structured(None, val_df)
    _, val_texts      = preprocess_text(None, val_df)
    _, y_val          = get_labels(None, val_df)

    # ── Load models ──────────────────────────────────────────────────────
    log.info("Loading models …")
    dtc    = DeepTriageCN.load_from_file(str(models_dir / "deeptriage_cn.pkl"))
    tabnet = TabNetWrapper(); tabnet.load(str(models_dir / "tabnet"))
    vit    = joblib.load(models_dir / "vitals_only_xgb.pkl")
    enc    = dtc.structured_encoder

    # Baseline AUROCs (no degradation)
    X_val_std_base = enc.transform(X_val_raw)
    val_emb        = dtc.text_encoder.encode(val_texts.tolist())
    X_fused_base   = np.hstack([val_emb, X_val_std_base])

    baseline = {
        "DeepTriage-CN":         roc_auc_score(y_val, dtc.predict_proba(val_texts.tolist(), X_val_raw)),
        "TabNet":                roc_auc_score(y_val, tabnet.predict_proba(X_val_raw)[:, 1]),
        "XGBoost (Vitals-only)": roc_auc_score(y_val, vit.predict_proba(X_val_std_base)[:, 1]),
    }
    log.info("Baseline AUROCs: %s", {k: f"{v:.4f}" for k,v in baseline.items()})

    # ── Simulation ──────────────────────────────────────────────────────
    rows = []
    for prop in proportions:
        log.info("Missing proportion: %.0f%% …", prop*100)
        for rep in range(n_reps):
            rng = np.random.default_rng(cfg["random_state"] + rep * 100)
            # Degrade validation structured features
            X_val_deg = _apply_degradation(X_val_raw, enc, prop, sigma_mult, rng)
            X_fused_deg = np.hstack([val_emb, X_val_deg])

            a_dtc = roc_auc_score(y_val, dtc.classifier.predict_proba(X_fused_deg)[:, 1])
            a_tab = roc_auc_score(y_val, tabnet.predict_proba_from_std(X_val_deg)[:, 1])
            a_vit = roc_auc_score(y_val, vit.predict_proba(X_val_deg)[:, 1])

            for model, auroc in [("DeepTriage-CN", a_dtc),
                                  ("TabNet",        a_tab),
                                  ("XGBoost (Vitals-only)", a_vit)]:
                rows.append({
                    "model":               model,
                    "missing_proportion":  prop,
                    "repetition":          rep,
                    "auroc":               auroc,
                    "auroc_baseline":      baseline[model],
                    "auroc_retention":     auroc / baseline[model],
                })
            log.info(
                "  rep %d: DTC=%.4f (%.1f%%), Tab=%.4f (%.1f%%), "
                "Vit=%.4f (%.1f%%)",
                rep,
                a_dtc, a_dtc/baseline["DeepTriage-CN"]*100,
                a_tab, a_tab/baseline["TabNet"]*100,
                a_vit, a_vit/baseline["XGBoost (Vitals-only)"]*100,
            )

    rob_df = pd.DataFrame(rows)
    out    = results_dir / "robustness_results.csv"
    rob_df.to_csv(out, index=False)
    log.info("Results saved → %s", out)

    # Print summary matching paper Section 4.5
    log.info("\nSummary at 30%% missingness:")
    for model in baseline:
        sub = rob_df[(rob_df["model"]==model) & (rob_df["missing_proportion"]==0.30)]
        log.info("  %s: AUROC=%.4f ± %.4f  retention=%.1f%%",
                 model, sub["auroc"].mean(), sub["auroc"].std(),
                 sub["auroc_retention"].mean()*100)


if __name__ == "__main__":
    main()
