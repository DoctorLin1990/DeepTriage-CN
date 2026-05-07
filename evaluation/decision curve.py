#!/usr/bin/env python3
"""
decision_curve.py

Implements decision curve analysis (DCA) for evaluating clinical net benefit
across a range of threshold probabilities (Section 4.3, Figure 4).

Net benefit formula:
    NB = (TP / N) - (FP / N) * (p_t / (1 - p_t))
where p_t is the threshold probability.
"""

import numpy as np
from typing import Tuple


def compute_net_benefit(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    thresholds: np.ndarray = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net benefit for a predictive model across thresholds.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Model predicted probabilities for class 1.
        thresholds: Array of threshold probabilities. If None, use np.linspace(0.01, 0.99, 99).

    Returns:
        thresholds: The thresholds used.
        net_benefit_model: Net benefit of the model at each threshold.
        net_benefit_all: Net benefit of treating all patients.
    """
    if thresholds is None:
        thresholds = np.linspace(0.01, 0.99, 99)

    n = len(y_true)
    net_benefit_model = np.zeros_like(thresholds)
    net_benefit_all = np.zeros_like(thresholds)

    for i, pt in enumerate(thresholds):
        y_pred = (y_pred_proba >= pt).astype(int)
        tp = np.sum((y_pred == 1) & (y_true == 1))
        fp = np.sum((y_pred == 1) & (y_true == 0))

        # Model net benefit
        net_benefit_model[i] = (tp / n) - (fp / n) * (pt / (1 - pt))

        # Treat all net benefit
        prevalence = np.mean(y_true)
        net_benefit_all[i] = prevalence - (1 - prevalence) * (pt / (1 - pt))

    return thresholds, net_benefit_model, net_benefit_all