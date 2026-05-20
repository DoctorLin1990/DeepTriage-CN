#!/usr/bin/env python3
"""
metrics.py

Binary classification metrics for Table 2 of the paper.

Metrics computed (Section 3.8):
    - AUROC  : Area under the ROC curve
    - AUPRC  : Area under the precision-recall curve
    - Sensitivity (recall) at the Youden-optimal threshold
    - Specificity at the Youden-optimal threshold
    - Accuracy at the Youden-optimal threshold
    - F1 score at the Youden-optimal threshold
    - Positive Predictive Value (PPV) at threshold
    - Negative Predictive Value (NPV) at threshold

The Youden-optimal threshold for DeepTriage-CN is 0.28 (Section 3.8).
All metrics are reported without artificial rounding (paper Section 4.2).
"""

import numpy as np
from typing import Dict, Any
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    accuracy_score,
)


def compute_all_binary_metrics(
    y_true: np.ndarray,
    y_score: np.ndarray,
    threshold: float = 0.28,
) -> Dict[str, Any]:
    """
    Compute all Table 2 metrics for a given model.

    Args:
        y_true    : Ground-truth binary labels (0 or 1).
        y_score   : Predicted probabilities in [0, 1].
        threshold : Decision threshold (default 0.28 = Youden-optimal for
                    DeepTriage-CN from training set, Section 3.8).

    Returns:
        Dict with keys:
            auroc, auprc, sensitivity, specificity, accuracy, f1, ppv, npv
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_score = np.asarray(y_score, dtype=float)

    # Ranking metrics (threshold-free)
    auroc = roc_auc_score(y_true, y_score)
    auprc = average_precision_score(y_true, y_score)

    # Binary predictions at the operating threshold
    y_pred = (y_score >= threshold).astype(int)

    # Confusion matrix: [[TN, FP], [FN, TP]]
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0   # recall
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv         = tp / (tp + fp) if (tp + fp) > 0 else 0.0   # precision
    npv         = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy    = accuracy_score(y_true, y_pred)
    f1          = f1_score(y_true, y_pred, zero_division=0)

    return {
        "auroc":       auroc,
        "auprc":       auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "accuracy":    accuracy,
        "f1":          f1,
        "ppv":         ppv,
        "npv":         npv,
    }
