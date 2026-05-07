#!/usr/bin/env python3
"""
calibration.py

Calibration assessment for probabilistic predictions, as described in
Section 4.3 of the paper.

Implements:
    - Brier score
    - Calibration intercept and slope (via logistic recalibration)
    - Hosmer-Lemeshow goodness-of-fit test
    - Calibration curve data for plotting (binned observed vs predicted)

These align with Figure 3 and the reported calibration statistics:
Brier = 0.12, slope = 0.95, intercept = 0.02, HL p = 0.28.
"""

import numpy as np
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from scipy.stats import chi2
from typing import Tuple, Dict


def brier_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute Brier score (mean squared error of probabilistic predictions)."""
    return np.mean((y_pred - y_true) ** 2)


def calibration_intercept_slope(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> Tuple[float, float]:
    """
    Compute calibration intercept and slope by regressing y_true on logit(y_pred).

    Perfect calibration: intercept = 0, slope = 1.

    Returns:
        intercept, slope
    """
    # Avoid logit(0) or logit(1) by clipping
    eps = 1e-12
    p = np.clip(y_pred, eps, 1 - eps)
    logit_p = np.log(p / (1 - p))

    # Logistic recalibration
    lr = LogisticRegression(penalty=None, solver='lbfgs', max_iter=1000)
    lr.fit(logit_p.reshape(-1, 1), y_true)
    intercept = lr.intercept_[0]
    slope = lr.coef_[0][0]
    return intercept, slope


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[float, float]:
    """
    Hosmer-Lemeshow goodness-of-fit test.

    Divides predictions into deciles and compares observed vs expected counts
    using a chi-squared statistic. Reported in Section 4.3.

    Returns:
        chi2_stat, p_value
    """
    bin_edges = np.percentile(y_pred, np.linspace(0, 100, n_bins + 1))
    bin_edges[0] = -0.01  # ensure all samples included
    bin_edges[-1] = 1.01

    observed_pos = np.zeros(n_bins)
    expected_pos = np.zeros(n_bins)
    observed_neg = np.zeros(n_bins)
    expected_neg = np.zeros(n_bins)

    for i in range(n_bins):
        mask = (y_pred > bin_edges[i]) & (y_pred <= bin_edges[i + 1])
        n_bin = np.sum(mask)
        if n_bin == 0:
            continue
        observed_pos[i] = np.sum(y_true[mask])
        observed_neg[i] = n_bin - observed_pos[i]
        expected_pos[i] = np.sum(y_pred[mask])
        expected_neg[i] = n_bin - expected_pos[i]

    # Avoid division by zero
    expected_pos = np.maximum(expected_pos, 1e-6)
    expected_neg = np.maximum(expected_neg, 1e-6)

    chi2_pos = (observed_pos - expected_pos) ** 2 / expected_pos
    chi2_neg = (observed_neg - expected_neg) ** 2 / expected_neg
    chi2_stat = np.sum(chi2_pos + chi2_neg)

    # Degrees of freedom: n_bins - 2
    df = n_bins - 2
    p_value = 1 - chi2.cdf(chi2_stat, df)
    return chi2_stat, p_value


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Dict[str, float]:
    """
    Compute all calibration metrics and return as a dictionary.

    Returns keys: 'brier', 'intercept', 'slope', 'hl_chi2', 'hl_pvalue'
    """
    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    hl_stat, hl_p = hosmer_lemeshow_test(y_true, y_pred, n_bins)
    brier = brier_score(y_true, y_pred)

    return {
        "brier": brier,
        "intercept": intercept,
        "slope": slope,
        "hl_chi2": hl_stat,
        "hl_pvalue": hl_p,
    }


def calibration_curve_data(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate binned calibration curve data (observed vs predicted proportions).

    Returns:
        fraction_of_positives, mean_predicted_value
    """
    fraction_pos, mean_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    return fraction_pos, mean_pred