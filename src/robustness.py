#!/usr/bin/env python3
"""
robustness.py

Data-degradation simulation described in Section 3.7 of the paper.

The simulation replicates clinically plausible missing-not-at-random (MNAR)
conditions: the probability of a vital sign being missing is modelled as an
inverse logistic decay function centred on the variable's normal physiological
range, so that near-normal readings are preferentially omitted (simulating
low-acuity nurse heuristics).

Additionally, Gaussian noise ε ~ N(0, σ_x · noise_sigma) is injected into
the remaining observed continuous values to simulate documentation noise.

Output: a dict mapping each missing_proportion → degraded raw feature array.

Paper reference  : Section 3.7, Supplementary Figure 1
Key numbers:
    missing_proportions : [0.10, 0.20, 0.30]   (10 %, 20 %, 30 %)
    noise_sigma         : 0.5  (multiplied by per-feature σ_x)
    Retention at 30 %   : DeepTriage-CN 95.4 %, TabNet 83.2 %, XGBoost 82.8 %
"""

import numpy as np
from typing import Dict, List, Optional
import logging

logger = logging.getLogger(__name__)

# Approximate midpoints of healthy physiological ranges for each feature.
# Used to centre the MNAR logistic decay (features near normal → higher P(missing)).
_PHYSIO_MIDPOINTS: Dict[str, float] = {
    "age":              50.0,   # not removed (demographic)
    "sex":               0.5,   # not removed (demographic)
    "temperature":      37.0,
    "heart_rate":       75.0,
    "respiratory_rate": 16.0,
    "sbp":             120.0,
    "dbp":              80.0,
    "spo2":             98.0,
}

# Features that should NOT be masked in the robustness simulation
# (demographics are always available at triage registration).
_NON_MASKABLE = {"age", "sex"}


def simulate_degraded_data(
    X_val_raw: np.ndarray,
    feature_names: List[str],
    missing_proportions: List[float] = (0.10, 0.20, 0.30),
    noise_sigma: float = 0.5,
    seed: int = 42,
) -> Dict[float, np.ndarray]:
    """
    Simulate MNAR data degradation with compounded Gaussian noise.

    For each proportion p in *missing_proportions*:
        1. Compute MNAR missingness weights via inverse logistic proximity
           to each feature's physiological midpoint (healthy readings are
           preferentially omitted).
        2. Sample NaN masks such that approximately p × n entries
           in the maskable columns are set to NaN.
        3. Inject Gaussian noise ε ~ N(0, σ_x · noise_sigma) into the
           remaining (non-NaN) continuous entries.

    Args:
        X_val_raw           : Raw validation features, shape (n, d).
                              May already contain NaN from real missingness.
        feature_names       : Column names in the same order as X_val_raw.
        missing_proportions : Proportions to simulate (paper: [0.10, 0.20, 0.30]).
        noise_sigma         : Gaussian noise multiplier (paper: 0.5).
        seed                : Base RNG seed; each proportion uses seed+i.

    Returns:
        Dict mapping each proportion float → degraded raw array (n, d).
    """
    rng = np.random.default_rng(seed)
    X = np.asarray(X_val_raw, dtype=float)
    n, d = X.shape

    # Identify maskable column indices (continuous vitals only)
    maskable_idx = [
        i for i, name in enumerate(feature_names)
        if name not in _NON_MASKABLE
    ]
    maskable_names = [feature_names[i] for i in maskable_idx]

    # Per-feature training standard deviations (estimated from val for noise scale)
    # Using nanstd avoids influence of existing NaN from real missingness.
    feat_std = np.nanstd(X, axis=0)
    feat_std[feat_std == 0] = 1.0  # guard against constant features

    results: Dict[float, np.ndarray] = {}

    for i, prop in enumerate(missing_proportions):
        rng_i = np.random.default_rng(seed + i)
        X_deg = X.copy()

        # --- Step 1: MNAR missingness ---
        for col_idx, col_name in zip(maskable_idx, maskable_names):
            col_vals = X_deg[:, col_idx]
            midpoint  = _PHYSIO_MIDPOINTS.get(col_name, np.nanmean(col_vals))
            feat_range = feat_std[col_idx] if feat_std[col_idx] > 0 else 1.0

            # Proximity score: higher when value is near the physiological midpoint
            proximity = np.exp(
                -0.5 * ((col_vals - midpoint) / feat_range) ** 2
            )
            # Replace NaN proximity (already missing) with 0 so they are not
            # double-counted in the missingness budget.
            proximity = np.nan_to_num(proximity, nan=0.0)

            # Normalise to get a probability distribution over rows
            total = proximity.sum()
            if total == 0:
                probs = np.ones(n) / n
            else:
                probs = proximity / total

            # Number of additional entries to mask in this column
            n_currently_missing = int(np.isnan(X_deg[:, col_idx]).sum())
            n_target_missing     = max(0, int(prop * n) - n_currently_missing)

            if n_target_missing > 0:
                # Draw rows to mask, weighted by proximity; no replacement
                available = np.where(~np.isnan(X_deg[:, col_idx]))[0]
                probs_available = probs[available]
                probs_available = probs_available / probs_available.sum()
                n_to_draw = min(n_target_missing, len(available))
                to_mask = rng_i.choice(
                    available, size=n_to_draw, replace=False, p=probs_available
                )
                X_deg[to_mask, col_idx] = np.nan

        # --- Step 2: Gaussian noise injection into remaining observed values ---
        for col_idx in maskable_idx:
            obs_mask = ~np.isnan(X_deg[:, col_idx])
            noise = rng_i.normal(
                loc=0.0,
                scale=noise_sigma * feat_std[col_idx],
                size=int(obs_mask.sum()),
            )
            X_deg[obs_mask, col_idx] += noise

        actual_missing_rate = np.isnan(X_deg[:, maskable_idx]).mean()
        logger.info(
            f"[Robustness] prop={prop:.0%} → actual masked rate "
            f"(maskable cols): {actual_missing_rate:.3f}"
        )
        results[prop] = X_deg

    return results
