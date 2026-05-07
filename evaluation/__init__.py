"""
Evaluation Package for DeepTriage-CN
=====================================
Implements all statistical and clinical evaluation metrics described in the paper.

Modules:
    metrics             - AUROC, AUPRC, sensitivity, specificity, accuracy
    calibration         - Brier score, calibration slope/intercept, Hosmer-Lemeshow
    decision_curve      - Decision curve analysis (net benefit)
    bootstrap           - Bootstrap confidence intervals for all metrics
    statistical_tests   - DeLong test, continuous NRI
    error_analysis      - Stratified false-positive/false-negative rates by age and narrative length
"""

from .metrics import compute_all_binary_metrics
from .calibration import compute_calibration_metrics
from .decision_curve import compute_net_benefit
from .bootstrap import bootstrap_confidence_intervals
from .statistical_tests import delong_test, continuous_nri
from .error_analysis import error_analysis_by_subgroup

__all__ = [
    "compute_all_binary_metrics",
    "compute_calibration_metrics",
    "compute_net_benefit",
    "bootstrap_confidence_intervals",
    "delong_test",
    "continuous_nri",
    "error_analysis_by_subgroup",
]