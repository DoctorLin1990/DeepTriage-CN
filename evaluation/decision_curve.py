#!/usr/bin/env python3
"""
decision_curve.py

Net benefit computation for Decision Curve Analysis (Figure 4, Section 4.3).

Paper: "Decision curve analysis quantified net benefit across risk thresholds."
       (Section 3.8)
       "DeepTriage-CN demonstrated a higher net benefit than 'treat all' or
       'treat none' strategies across a threshold range of approximately 15%
       to 75%." (Section 4.3)

Net benefit formula:
    NB(pt) = TP/N − FP/N × pt/(1−pt)

where pt is the threshold probability, N is the total sample size,
and TP/FP are computed at that threshold.

NOTE: This module uses an ABSOLUTE import path:
    from evaluation.decision_curve import compute_net_benefit

This is intentional — 'evaluation' and 'visualization' are sibling packages
under the project root (not parent-child). Relative imports from
visualization/ to evaluation/ would ascend beyond the package boundary and
raise ImportError. All entry-point scripts add the project root to sys.path,
so the absolute import resolves correctly.
"""

import numpy as np
from typing import Tuple


def compute_net_benefit(
    y_true: np.ndarray,
    y_score: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute net benefit for a model and the 'treat all' reference strategy.

    Args:
        y_true     : Ground-truth binary labels (0 or 1).
        y_score    : Predicted probabilities in [0, 1].
        thresholds : Array of threshold probabilities to evaluate.

    Returns:
        Tuple of three arrays (same length as thresholds):
            thresholds    : Echo of the input threshold array.
            nb_model      : Net benefit of the model at each threshold.
            nb_treat_all  : Net benefit of the 'treat all' strategy.
    """
    y_true  = np.asarray(y_true,  dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    n       = len(y_true)
    prev    = y_true.mean()   # event prevalence

    nb_model     = np.zeros_like(thresholds, dtype=float)
    nb_treat_all = np.zeros_like(thresholds, dtype=float)

    for i, pt in enumerate(thresholds):
        if pt <= 0 or pt >= 1:
            continue

        # Binary predictions at this threshold
        y_pred = (y_score >= pt).astype(float)
        tp     = np.sum((y_pred == 1) & (y_true == 1))
        fp     = np.sum((y_pred == 1) & (y_true == 0))

        nb_model[i]     = tp / n - fp / n * pt / (1 - pt)

        # 'Treat all' strategy: all patients receive intervention
        tp_all          = prev * n
        fp_all          = (1 - prev) * n
        nb_treat_all[i] = tp_all / n - fp_all / n * pt / (1 - pt)

    # Set negative net benefits to zero for presentation (dominated by 'treat none')
    nb_model     = np.maximum(nb_model, 0)

    return thresholds, nb_model, nb_treat_all
