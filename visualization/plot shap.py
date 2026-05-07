#!/usr/bin/env python3
"""
plot_shap.py

Reproduces Figure 5: SHAP beeswarm summary plot for DeepTriage-CN.
We compute SHAP values for the XGBoost classifier using the full feature matrix
(776 dimensions). The plot is simplified to show top features including
text-derived features and vital signs.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
import os


def plot_shap_beeswarm(
    model,  # Trained XGBoostClassifier or its booster
    X_fused: np.ndarray,
    feature_names: list,
    max_display: int = 20,
    save_path: str = "outputs/figures/figure5_shap.tiff",
    figsize: tuple = (10, 8),
    dpi: int = 300,
):
    """
    Generate SHAP beeswarm summary plot for the DeepTriage-CN framework.

    Args:
        model: Trained XGBoost model (must have .get_booster() if XGBClassifier).
        X_fused: Fused feature matrix (n_samples, 776).
        feature_names: List of feature names of length 776.
        max_display: Maximum number of features to show.
        save_path: Save path.
        figsize: Figure size.
        dpi: Resolution.
    """
    # Create explainer
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_fused)

    # For binary classification, shap_values is (n_samples, n_features) for the positive class
    if isinstance(shap_values, list):
        shap_values = shap_values[1]

    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    shap.summary_plot(
        shap_values,
        X_fused,
        feature_names=feature_names,
        max_display=max_display,
        show=False,
        plot_size=None,
        color_bar=True,
    )
    ax.set_title("SHAP Feature Importance for DeepTriage-CN", fontsize=14, fontweight="bold")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 5 saved to {save_path}")