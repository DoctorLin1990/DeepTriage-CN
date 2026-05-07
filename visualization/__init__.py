"""
Visualization Package for DeepTriage-CN
=======================================
Functions to reproduce all figures from the paper and supplementary materials.

Modules:
    plot_roc            - Figure 2: ROC curves for all models (overall and geriatric)
    plot_calibration    - Figure 3: Calibration plot with density sub-panel
    plot_decision_curve - Figure 4: Decision curve analysis
    plot_robustness     - Supplementary Figure 1: Robustness under data degradation
    plot_shap           - Figure 5: SHAP beeswarm summary plot
    plot_error_analysis - Figure 6: Error analysis bar charts
"""

from .plot_roc import plot_roc_overall, plot_roc_geriatric
from .plot_calibration import plot_calibration_curve
from .plot_decision_curve import plot_decision_curve
from .plot_robustness import plot_robustness_bars
from .plot_shap import plot_shap_beeswarm
from .plot_error_analysis import plot_error_distribution, plot_fnr_by_subgroup

__all__ = [
    "plot_roc_overall",
    "plot_roc_geriatric",
    "plot_calibration_curve",
    "plot_decision_curve",
    "plot_robustness_bars",
    "plot_shap_beeswarm",
    "plot_error_distribution",
    "plot_fnr_by_subgroup",
]