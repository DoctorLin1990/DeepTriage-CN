#!/usr/bin/env python3
"""
plot_shap.py

Generates Figure 5: SHAP beeswarm summary plot for DeepTriage-CN.

Paper Section 4.6 / Figure 5 caption:
    "SHAP analysis identified oxygen saturation, age, and text-derived features
     related to 'chest pain', 'dyspnea', and 'altered mental status' as the
     most influential variables."
    "Note: these post-hoc attributions represent pure mathematical associations
     within the algorithm's learned architecture and do not confer physiological
     causality or direct bedside actionability."

Bug L4 fix (from AUDIT_REPORT.md):
    The original generate_all_figures.py created a fresh StructuredEncoder
    and called fit_transform() on validation data, causing data leakage.
    The caller (generate_all_figures.py) MUST use:
        X_val_std_for_shap = deeptriage.structured_encoder.transform(X_val_raw)
    to standardise with training-set-derived parameters, then pass the fused
    array to this function. This module itself does not perform standardisation.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import List, Optional


def plot_shap_beeswarm(
    classifier,
    X_fused: np.ndarray,
    feature_names: List[str],
    max_display: int = 20,
    save_path: str = "outputs/figures/figure5_shap.tiff",
    dpi: int = 300,
) -> None:
    """
    Compute SHAP values for the XGBoost classifier and plot a beeswarm summary.

    Args:
        classifier    : Fitted xgb.XGBClassifier (DeepTriageCN.classifier).
        X_fused       : Fused feature matrix (n, 776) — BERT embeddings
                        concatenated with training-standardised vitals.
                        Must NOT be double-standardised (see Bug L4 note above).
        feature_names : List of 776 feature names (768 text_* + 8 structured).
        max_display   : Number of top features to show in beeswarm (default 20).
        save_path     : Output TIFF path.
        dpi           : Resolution.
    """
    try:
        import shap
    except ImportError as e:
        raise ImportError(
            "The 'shap' package is required.  Install with: pip install shap"
        ) from e

    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
    })

    explainer   = shap.TreeExplainer(classifier)
    shap_values = explainer.shap_values(X_fused)

    # shap_values may be a list [class0, class1] for binary XGBoost
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Create beeswarm figure
    fig, ax = plt.subplots(figsize=(6, 5))
    shap.summary_plot(
        shap_values,
        X_fused,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_type="dot",
    )
    plt.title(
        "SHAP Feature Attribution — DeepTriage-CN\n"
        "(post-hoc mathematical associations only; not causal evidence)",
        fontsize=7, pad=6,
    )

    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    plt.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close("all")
    print(f"Figure 5 saved → {save_path}")
