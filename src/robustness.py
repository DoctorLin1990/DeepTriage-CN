#!/usr/bin/env python3
"""
robustness.py

Implements the data degradation simulation described in Section 3.7
("Robustness Testing") and reported in Section 4.5.

Two types of perturbations are applied:
    1. Missing-not-at-random (MNAR) omission: The probability of a variable
       being missing P(M|X) is modeled as an inverse logistic decay centered
       on the normal physiological range (higher chance of missing when closer
       to normal). For example, a normal respiratory rate (16) is more likely
       to be omitted than an abnormal one.
    2. Gaussian noise injection: N(0, 0.5 * σ_x) added to the remaining
       continuous variables.

The simulation is repeated for aggregate missing proportions of 10%, 20%,
and 30%, as reported in the paper.

NOTE (from Discussion): This simulation has a structured missingness pattern
that may not fully represent real-world heterogeneous documentation failures.
The results should be interpreted with appropriate caveats (Section 5).
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Dict
from scipy.special import expit  # logistic function
import logging

logger = logging.getLogger(__name__)

# Normal physiological ranges for each vital sign (approximate midpoints)
NORMAL_RANGES = {
    "temperature": 36.8,
    "heart_rate": 78,
    "respiratory_rate": 16,
    "sbp": 125,
    "dbp": 78,
    "spo2": 98,
}


def mnar_missingness(
    X: np.ndarray,
    feature_names: List[str],
    missing_proportion: float,
    normal_values: Dict[str, float] = NORMAL_RANGES,
) -> np.ndarray:
    """
    Apply MNAR missingness to vital sign data.

    The probability of a feature being missing is higher when the value
    is close to its normal physiological baseline, simulating the clinical
    heuristic of omitting measurements on apparently stable patients.

    Formula: P(M|X) = σ(-α * |x - μ_normal| + β)
    where α controls steepness, β shifts the threshold to achieve the
    target missing proportion.

    Args:
        X: Array of shape (n_samples, n_features), containing raw values.
        feature_names: Names of the columns (must include those in normal_values).
        missing_proportion: Target proportion of values to be set to NaN (0-1).
        normal_values: Dictionary mapping feature name to its normal value.

    Returns:
        Modified X with NaN values injected.
    """
    X_mod = X.copy()
    n_samples, n_features = X.shape

    for col_idx, col_name in enumerate(feature_names):
        if col_name not in normal_values:
            continue

        col_vals = X[:, col_idx]
        # Compute distance from normal
        normal_val = normal_values[col_name]
        # Robust scaling: use MAD (median absolute deviation) as reference spread
        # to avoid extreme scales
        valid_mask = ~np.isnan(col_vals)
        if valid_mask.sum() == 0:
            continue

        spread = np.nanstd(col_vals)
        if spread == 0:
            spread = 1.0
        dist = np.abs(col_vals - normal_val) / spread

        # Inverse logistic decay: higher prob when dist is small
        # We calibrate β to reach the desired missing rate
        # Start with a fixed slope α = 2.0 (reasonable decay)
        alpha = 2.0
        # Find β such that mean probability ≈ missing_proportion for this column
        # Use binary search
        beta_low, beta_high = -10.0, 10.0
        for _ in range(30):
            beta_mid = (beta_low + beta_high) / 2
            prob = expit(-alpha * dist + beta_mid)
            mean_prob = np.nanmean(prob)
            if mean_prob > missing_proportion:
                beta_high = beta_mid
            else:
                beta_low = beta_mid

        prob = expit(-alpha * dist + beta_low)
        # Clip probabilities to avoid extreme values
        prob = np.clip(prob, 0.0, 1.0)
        # Randomly mask
        mask = np.random.random(n_samples) < prob
        X_mod[mask, col_idx] = np.nan

    return X_mod


def inject_gaussian_noise(
    X: np.ndarray,
    sigma_multiplier: float = 0.5,
    feature_names: List[str] = None
) -> np.ndarray:
    """
    Add Gaussian noise to continuous variables.

    Noise ~ N(0, sigma_multiplier * σ_feature)
    where σ_feature is the standard deviation of the feature (computed on
    non-missing values).

    Args:
        X: Array of shape (n_samples, n_features). May contain NaN.
        sigma_multiplier: Multiplier for feature std (paper: 0.5).
        feature_names: Ignored; kept for interface compatibility.

    Returns:
        X with noise added to non-NaN entries.
    """
    X_noisy = X.copy()
    n_features = X.shape[1]

    for j in range(n_features):
        col = X[:, j]
        valid = ~np.isnan(col)
        if valid.sum() < 2:
            continue
        sigma = np.nanstd(col) * sigma_multiplier
        noise = np.random.normal(0, sigma, size=X.shape[0])
        X_noisy[valid, j] = col[valid] + noise[valid]

    return X_noisy


def simulate_degraded_data(
    X: np.ndarray,
    feature_names: List[str],
    missing_proportions: List[float] = [0.1, 0.2, 0.3],
    noise_sigma: float = 0.5,
    seed: int = 42,
) -> Dict[float, np.ndarray]:
    """
    Generate multiple degraded versions of the dataset for robustness testing.

    For each missing proportion, apply MNAR missingness and then Gaussian noise.

    Args:
        X: Original clean feature matrix.
        feature_names: Names of features corresponding to columns.
        missing_proportions: List of target missing fractions (e.g., 0.1, 0.2, 0.3).
        noise_sigma: Sigma multiplier for noise injection.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary mapping missing proportion -> degraded feature matrix.
    """
    np.random.seed(seed)
    degraded_sets = {}
    for prop in missing_proportions:
        logger.info(f"Simulating missing proportion = {prop:.1%}...")
        X_miss = mnar_missingness(X, feature_names, prop)
        X_degraded = inject_gaussian_noise(X_miss, noise_sigma, feature_names)
        degraded_sets[prop] = X_degraded

    return degraded_sets