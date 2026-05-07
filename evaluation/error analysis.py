#!/usr/bin/env python3
"""
error_analysis.py

Stratified error analysis as described in Section 4.7 and Figure 6.

Key findings from the paper:
    - False negatives: disproportionately among encounters with very brief
      chief complaints (≤3 words), e.g., "dizziness".
    - False-negative rate in older patients with sparse narratives: 6.1%
      vs 3.2% in younger counterparts.
    - False positives: frequently involve young patients with acute benign pain
      and transient vital sign elevations.

This module provides functions to compute these subgroup error rates.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple


def error_analysis_by_subgroup(
    y_true: np.ndarray,
    y_pred_binary: np.ndarray,
    ages: np.ndarray,
    complaint_lengths: np.ndarray,
    age_threshold: int = 65,
    sparse_narrative_threshold: int = 3,
) -> Dict[str, Dict[str, float]]:
    """
    Perform stratified error analysis.

    Args:
        y_true: Ground truth labels (0/1).
        y_pred_binary: Model binary predictions at the Youden threshold.
        ages: Patient ages (years).
        complaint_lengths: Number of words/tokens in chief complaint.
        age_threshold: Cutoff for "older" adults (65).
        sparse_narrative_threshold: Max words to consider narrative "sparse" (≤3).

    Returns:
        Nested dictionary with error rates by subgroup.
        {
            "overall": {"fpr": ..., "fnr": ..., "accuracy": ...},
            "young_sparse": {"fnr": ..., "n": ...},
            "elder_sparse": {"fnr": ..., "n": ...},
            ...
        }
    """
    # Masks
    elder = ages >= age_threshold
    young = ~elder
    sparse = complaint_lengths <= sparse_narrative_threshold

    # Confusion matrix helpers
    fn = (y_true == 1) & (y_pred_binary == 0)  # missed admissions
    fp = (y_true == 0) & (y_pred_binary == 1)  # unnecessary alerts

    n_total = len(y_true)

    # Overall error rates
    overall_fpr = np.sum(fp) / n_total
    overall_fnr = np.sum(fn) / n_total
    overall_acc = np.mean(y_true == y_pred_binary)

    # Subgroup false-negative rates (percentage of true admissions missed)
    def compute_fnr(subgroup_mask):
        n_admitted = np.sum(y_true[subgroup_mask] == 1)
        if n_admitted == 0:
            return 0.0, 0
        n_missed = np.sum(fn[subgroup_mask])
        return n_missed / n_admitted, n_admitted

    young_sparse_fnr, young_sparse_n = compute_fnr(young & sparse)
    elder_sparse_fnr, elder_sparse_n = compute_fnr(elder & sparse)
    young_rich_fnr, young_rich_n = compute_fnr(young & ~sparse)
    elder_rich_fnr, elder_rich_n = compute_fnr(elder & ~sparse)

    # False positive rates in young patients with rich text
    def compute_fpr(subgroup_mask):
        n_discharged = np.sum(y_true[subgroup_mask] == 0)
        if n_discharged == 0:
            return 0.0, 0
        n_false_alarm = np.sum(fp[subgroup_mask])
        return n_false_alarm / n_discharged, n_discharged

    young_rich_fpr, young_rich_n_dis = compute_fpr(young & ~sparse)

    results = {
        "overall": {
            "fpr": overall_fpr,
            "fnr": overall_fnr,
            "accuracy": overall_acc,
            "n": n_total,
        },
        "young_sparse": {"fnr": young_sparse_fnr, "n": young_sparse_n},
        "elder_sparse": {"fnr": elder_sparse_fnr, "n": elder_sparse_n},
        "young_rich": {"fnr": young_rich_fnr, "n": young_rich_n},
        "elder_rich": {"fnr": elder_rich_fnr, "n": elder_rich_n},
        "young_rich_fpr": {"fpr": young_rich_fpr, "n": young_rich_n_dis},
    }
    return results