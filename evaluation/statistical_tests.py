#!/usr/bin/env python3
"""
statistical_tests.py

Pairwise AUROC comparisons (DeLong's test) and continuous Net Reclassification
Improvement (NRI), as described in Section 3.8 of the paper.

DeLong test:
    Pairwise comparison of AUROCs between DeepTriage-CN and each comparator.
    Implemented via the fast covariance method of DeLong et al. (1988).

Continuous NRI:
    "Continuous net reclassification improvement was calculated relative to
    both NEWS2 and ESI." (Section 3.8)
    Values reported: NRI vs NEWS2 = 0.41 (95 % CI 0.34–0.48);
                     NRI vs ESI   = 0.53 (95 % CI 0.46–0.60).

References:
    DeLong ER, DeLong DM, Clarke-Pearson DL. Comparing the areas under two or
    more correlated receiver operating characteristic curves: a nonparametric
    approach. Biometrics. 1988;44(3):837–845.

    Pencina MJ, et al. Evaluating the added predictive ability of a new marker:
    from area under the ROC curve to reclassification and beyond.
    Stat Med. 2008;27(2):157–172.
"""

import numpy as np
from scipy import stats
from typing import Tuple


# ---------------------------------------------------------------------------
# DeLong test
# ---------------------------------------------------------------------------

def _placement_values(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute placement values V10 (controls classified below a case)
    and V01 (cases classified above a control).

    These are the building blocks for DeLong's covariance estimator.
    """
    cases    = y_score[y_true == 1]
    controls = y_score[y_true == 0]
    m = len(cases)
    r = len(controls)

    # V10[i] = (1/r) Σ_j I(cases[i] > controls[j]) + 0.5·I(cases[i] == controls[j])
    V10 = np.array([
        np.mean(
            (c > controls).astype(float) + 0.5 * (c == controls).astype(float)
        )
        for c in cases
    ])
    # V01[j] = (1/m) Σ_i I(cases[i] > controls[j]) + 0.5·I(cases[i] == controls[j])
    V01 = np.array([
        np.mean(
            (cases > ctrl).astype(float) + 0.5 * (cases == ctrl).astype(float)
        )
        for ctrl in controls
    ])
    return V10, V01


def delong_test(
    y_true: np.ndarray,
    y_score_a: np.ndarray,
    y_score_b: np.ndarray,
) -> Tuple[float, float]:
    """
    DeLong's test for the difference between two correlated AUROCs.

    Args:
        y_true    : Ground-truth binary labels.
        y_score_a : Predicted probabilities for model A.
        y_score_b : Predicted probabilities for model B.

    Returns:
        (z_statistic, two-sided p_value)
    """
    y_true    = np.asarray(y_true,    dtype=int)
    y_score_a = np.asarray(y_score_a, dtype=float)
    y_score_b = np.asarray(y_score_b, dtype=float)

    m = int(np.sum(y_true == 1))   # number of cases
    r = int(np.sum(y_true == 0))   # number of controls

    V10_a, V01_a = _placement_values(y_true, y_score_a)
    V10_b, V01_b = _placement_values(y_true, y_score_b)

    auc_a = V10_a.mean()
    auc_b = V10_b.mean()

    # Structural components for covariance matrix
    S10 = np.cov(np.vstack([V10_a, V10_b]))  # 2×2
    S01 = np.cov(np.vstack([V01_a, V01_b]))  # 2×2

    # Joint covariance matrix of (AUC_a, AUC_b)
    S = S10 / m + S01 / r   # 2×2

    # Variance of (AUC_a - AUC_b) = Var(AUC_a) + Var(AUC_b) - 2·Cov(AUC_a, AUC_b)
    var_diff = S[0, 0] + S[1, 1] - 2 * S[0, 1]
    if var_diff <= 0:
        return 0.0, 1.0

    z = (auc_a - auc_b) / np.sqrt(var_diff)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return float(z), float(p)


# ---------------------------------------------------------------------------
# Continuous NRI
# ---------------------------------------------------------------------------

def continuous_nri(
    y_true: np.ndarray,
    y_score_ref: np.ndarray,
    y_score_new: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Compute continuous (category-free) Net Reclassification Improvement.

    NRI_continuous = P(↑ | event) − P(↓ | event)
                   + P(↓ | non-event) − P(↑ | non-event)

    where ↑ means the new model assigns a higher probability than the reference,
    and ↓ means lower.

    Args:
        y_true       : Ground-truth binary labels.
        y_score_ref  : Reference model probabilities (e.g., NEWS2 / 20).
        y_score_new  : New model probabilities (DeepTriage-CN).

    Returns:
        (nri_total, nri_events, nri_nonevents)
    """
    y_true       = np.asarray(y_true,       dtype=int)
    y_score_ref  = np.asarray(y_score_ref,  dtype=float)
    y_score_new  = np.asarray(y_score_new,  dtype=float)

    delta = y_score_new - y_score_ref

    # Event group (admitted)
    ev_mask       = y_true == 1
    p_up_events   = np.mean(delta[ev_mask] > 0)
    p_down_events = np.mean(delta[ev_mask] < 0)
    nri_events    = p_up_events - p_down_events

    # Non-event group (discharged)
    nev_mask         = y_true == 0
    p_up_nonevents   = np.mean(delta[nev_mask] > 0)
    p_down_nonevents = np.mean(delta[nev_mask] < 0)
    nri_nonevents    = p_down_nonevents - p_up_nonevents

    nri_total = nri_events + nri_nonevents
    return float(nri_total), float(nri_events), float(nri_nonevents)
