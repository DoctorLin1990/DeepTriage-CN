#!/usr/bin/env python3
"""
metrics.py

Binary classification metrics for the DeepTriage-CN project.
Implements AUROC, AUPRC, sensitivity, specificity, accuracy, and PPV as
reported in Table 2 and Section 4.2.

All metrics are computed at a given probability threshold.
The Youden-optimal threshold (= 0.28) is used in the paper.
"""

import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
)
from typing import Tuple, Dict


def compute_all_binary_metrics(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float = 0.28,
) -> Dict[str, float]:
    """
    Compute a comprehensive set of binary classification metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_pred_proba: Predicted probabilities for class 1.
        threshold: Decision threshold for binary classification.

    Returns:
        Dictionary with keys:
            'auroc', 'auprc', 'sensitivity', 'specificity',
            'ppv', 'npv', 'accuracy', 'f1', 'threshold'
    """
    y_pred = (y_pred_proba >= threshold).astype(int)

    # AUROC and AUPRC (independent of threshold)
    auroc = roc_auc_score(y_true, y_pred_proba)
    auprc = average_precision_score(y_true, y_pred_proba)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0

    return {
        "auroc": auroc,
        "auprc": auprc,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "ppv": ppv,
        "npv": npv,
        "accuracy": accuracy,
        "f1": f1,
        "threshold": threshold,
    }


def find_youden_threshold(y_true: np.ndarray, y_pred_proba: np.ndarray) -> float:
    """
    Compute the Youden-optimal threshold (maximizes sensitivity + specificity - 1).

    This threshold is computed on the training set and fixed for validation
    (Section 3.8). In the paper, this value is 0.28 for DeepTriage-CN.

    Args:
        y_true: Ground truth labels.
        y_pred_proba: Predicted probabilities.

    Returns:
        Optimal threshold value.
    """
    from sklearn.metrics import roc_curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    youden_index = tpr - fpr
    optimal_idx = np.argmax(youden_index)
    return thresholds[optimal_idx]