#!/usr/bin/env python3
"""
bootstrap.py

Bootstrap confidence interval computation for evaluation metrics,
as described in Section 3.8.  The paper uses 1,000 bootstrap resamples
to estimate 95% CIs for AUROC, AUPRC, etc. (Table 2).
"""

import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from typing import Dict, Callable, Tuple


def bootstrap_metric(
    y_true: np.ndarray,
    y_score: np.ndarray,
    metric_fn: Callable,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    random_seed: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a single metric.

    Args:
        y_true: Ground truth labels.
        y_score: Predicted scores/probabilities.
        metric_fn: Function with signature f(y_true, y_score) -> float.
        n_bootstrap: Number of bootstrap iterations (paper: 1000).
        alpha: Significance level for CI (0.05 → 95% CI).
        random_seed: Seed for reproducibility.

    Returns:
        point_estimate, lower_ci, upper_ci
    """
    rng = np.random.RandomState(random_seed)
    n = len(y_true)
    indices = np.arange(n)
    estimates = np.zeros(n_bootstrap)

    for b in range(n_bootstrap):
        boot_idx = rng.choice(indices, size=n, replace=True)
        try:
            estimates[b] = metric_fn(y_true[boot_idx], y_score[boot_idx])
        except ValueError:
            # In case only one class present after resampling
            estimates[b] = np.nan

    # Filter out NaN values
    estimates = estimates[~np.isnan(estimates)]
    point_est = metric_fn(y_true, y_score)
    lower = np.percentile(estimates, 100 * alpha / 2)
    upper = np.percentile(estimates, 100 * (1 - alpha / 2))

    return point_est, lower, upper


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    random_seed: int = 42,
) -> Dict[str, Tuple[float, float, float]]:
    """
    Compute bootstrap CIs for AUROC and AUPRC (primary discrimination metrics).

    Returns:
        Dictionary mapping metric name → (point_est, lower, upper)
    """
    results = {}
    for name, fn in [("auroc", roc_auc_score), ("auprc", average_precision_score)]:
        point, lower, upper = bootstrap_metric(
            y_true, y_score, fn, n_bootstrap, random_seed=random_seed
        )
        results[name] = (point, lower, upper)
    return results