#!/usr/bin/env python3
"""
bootstrap.py

Bootstrap 95 % confidence intervals for AUROC, AUPRC, sensitivity,
specificity, and accuracy.

Paper reference: Section 3.8 — "Confidence intervals (95%) were obtained by
bootstrap resampling (1,000 iterations). All metric values are reported as
directly computed without artificial rounding; minor asymmetries in confidence
intervals reflect the inherent variability of the bootstrap distribution."
"""

import numpy as np
from typing import Dict
from .metrics import compute_all_binary_metrics


def bootstrap_confidence_intervals(
    y_true: np.ndarray,
    y_score: np.ndarray,
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
    threshold: float = 0.28,
    random_seed: int = 42,
) -> Dict[str, str]:
    """
    Compute 95 % CIs for each metric via percentile bootstrap.

    Args:
        y_true       : Ground-truth binary labels.
        y_score      : Predicted probabilities.
        n_bootstrap  : Number of resampling iterations (paper: 1,000).
        alpha        : Two-sided significance level (default 0.05 → 95 % CI).
        threshold    : Classification threshold for binary metrics.
        random_seed  : Reproducibility seed.

    Returns:
        Dict mapping metric name → "point_estimate (lower–upper)" string,
        matching the format reported in Table 2.
    """
    rng     = np.random.default_rng(random_seed)
    y_true  = np.asarray(y_true)
    y_score = np.asarray(y_score)
    n       = len(y_true)

    # --- Point estimates ---
    point = compute_all_binary_metrics(y_true, y_score, threshold)

    # --- Bootstrap distribution ---
    boot_records = {k: [] for k in point}
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        yt  = y_true[idx]
        ys  = y_score[idx]
        # Skip degenerate samples (only one class)
        if len(np.unique(yt)) < 2:
            continue
        m = compute_all_binary_metrics(yt, ys, threshold)
        for k, v in m.items():
            boot_records[k].append(v)

    lo = alpha / 2
    hi = 1.0 - alpha / 2

    result: Dict[str, str] = {}
    for metric, pt_val in point.items():
        arr = np.array(boot_records[metric])
        lower = np.quantile(arr, lo)
        upper = np.quantile(arr, hi)
        result[metric] = f"{pt_val:.3f} ({lower:.3f}–{upper:.3f})"

    return result
