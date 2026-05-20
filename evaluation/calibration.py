#!/usr/bin/env python3
"""
calibration.py

Calibration metrics for DeepTriage-CN, matching Section 4.3 of the paper:

    Brier score       : 0.12
    Calibration slope : 0.95
    Intercept         : 0.02
    Hosmer–Lemeshow   : p = 0.28

Reference: Section 3.8 — "Calibration was evaluated with the Brier score,
calibration intercept and slope, and the Hosmer–Lemeshow test."
"""

import numpy as np
from scipy import stats
from typing import Dict, Any


def brier_score(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute Brier score: mean squared difference between predicted probability
    and observed binary outcome.

    Lower is better; perfect model = 0.0; uninformative model ≈ p(1−p).

    Args:
        y_true  : Binary ground-truth labels.
        y_score : Predicted probabilities.

    Returns:
        Brier score (float).
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    return float(np.mean((y_score - y_true) ** 2))


def calibration_slope_intercept(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, float]:
    """
    Estimate calibration slope and intercept via logistic regression of
    observed outcomes on log-odds of predicted probabilities.

    A perfectly calibrated model has slope = 1.0 and intercept = 0.0.
    Paper reports: slope = 0.95, intercept = 0.02 (Section 4.3).

    Args:
        y_true  : Binary ground-truth labels.
        y_score : Predicted probabilities (clipped to avoid log(0)).

    Returns:
        Dict with keys 'slope' and 'intercept'.
    """
    from sklearn.linear_model import LogisticRegression

    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.clip(np.asarray(y_score, dtype=float), 1e-6, 1 - 1e-6)

    # Log-odds of predicted probability
    log_odds = np.log(y_score / (1 - y_score)).reshape(-1, 1)

    lr = LogisticRegression(fit_intercept=True, max_iter=1000, C=1e6)
    lr.fit(log_odds, y_true.astype(int))

    return {
        "slope":     float(lr.coef_[0][0]),
        "intercept": float(lr.intercept_[0]),
    }


def hosmer_lemeshow_test(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_groups: int = 10,
) -> Dict[str, float]:
    """
    Hosmer–Lemeshow goodness-of-fit test.

    Patients are divided into deciles of predicted risk; a chi-square statistic
    tests whether observed and expected event rates differ significantly across
    groups.  p > 0.05 indicates satisfactory calibration.

    Paper reports: Hosmer–Lemeshow p = 0.28 (Section 4.3).

    Args:
        y_true   : Binary ground-truth labels.
        y_score  : Predicted probabilities.
        n_groups : Number of equal-frequency bins (default 10 = deciles).

    Returns:
        Dict with keys 'hl_statistic' and 'p_value'.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n       = len(y_true)

    # Sort by predicted probability
    order   = np.argsort(y_score)
    y_true  = y_true[order]
    y_score = y_score[order]

    # Split into equal-frequency groups
    group_indices = np.array_split(np.arange(n), n_groups)
    hl_stat = 0.0

    for idx in group_indices:
        obs_events    = y_true[idx].sum()
        obs_nonevents = len(idx) - obs_events
        exp_events    = y_score[idx].sum()
        exp_nonevents = len(idx) - exp_events

        if exp_events > 0:
            hl_stat += (obs_events - exp_events) ** 2 / exp_events
        if exp_nonevents > 0:
            hl_stat += (obs_nonevents - exp_nonevents) ** 2 / exp_nonevents

    # Degrees of freedom = n_groups − 2
    df      = max(n_groups - 2, 1)
    p_value = 1 - stats.chi2.cdf(hl_stat, df=df)

    return {"hl_statistic": float(hl_stat), "p_value": float(p_value)}


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Dict[str, Any]:
    """
    Aggregate all calibration metrics used in Section 4.3.

    Returns:
        Dict with keys:
            brier_score, slope, intercept, hl_statistic, hl_p_value
    """
    bs  = brier_score(y_true, y_score)
    si  = calibration_slope_intercept(y_true, y_score)
    hl  = hosmer_lemeshow_test(y_true, y_score)

    return {
        "brier_score":  bs,
        "slope":        si["slope"],
        "intercept":    si["intercept"],
        "hl_statistic": hl["hl_statistic"],
        "hl_p_value":   hl["p_value"],
    }
