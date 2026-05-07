#!/usr/bin/env python3
"""
plot_roc.py

Reproduces Figure 2 from the DeepTriage-CN paper:
    (a) Overall validation set ROC curves for all models.
    (b) Geriatric subgroup (age ≥ 65) ROC curves.

Models are plotted as solid lines (ML architectures) or dashed lines
(conventional scores). The diagonal reference line is dotted.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from typing import Dict, List, Optional
import os


# Color and style settings consistent with the paper
MODEL_STYLES = {
    "DeepTriage-CN": {"color": "#E34A33", "linestyle": "-", "linewidth": 2.0, "label": "DeepTriage-CN (Multimodal)"},
    "TabNet": {"color": "#2C3E50", "linestyle": "-", "linewidth": 2.0, "label": "TabNet (Structured-only DL)"},
    "XGBoost (Vitals-only)": {"color": "#3498DB", "linestyle": "-", "linewidth": 2.0, "label": "XGBoost (Vitals-only)"},
    "Random Forest": {"color": "#95A5A6", "linestyle": "-", "linewidth": 1.8, "label": "Random Forest"},
    "Text-Only (BERT+LR)": {"color": "#F39C12", "linestyle": "-", "linewidth": 1.8, "label": "Text-Only (BERT + LR)"},
    "NEWS2": {"color": "#8E44AD", "linestyle": "--", "linewidth": 1.8, "label": "NEWS2"},
    "MEWS": {"color": "#16A085", "linestyle": "--", "linewidth": 1.8, "label": "MEWS"},
    "ESI": {"color": "#D35400", "linestyle": "--", "linewidth": 1.8, "label": "ESI"},
}


def plot_roc_overall(
    roc_data: Dict[str, Dict[str, np.ndarray]],
    save_path: str = "outputs/figures/figure2a_roc_overall.tiff",
    figsize: tuple = (8, 6),
    dpi: int = 300,
):
    """
    Plot ROC curves for all models on the overall temporal validation set (Figure 2a).

    Args:
        roc_data: Dictionary mapping model_name -> {"fpr": array, "tpr": array, "auroc": float}.
        save_path: Path to save the figure.
        figsize: Figure dimensions.
        dpi: Resolution.
    """
    fig, ax = plt.subplots(figsize=figsize)

    for model_name, style in MODEL_STYLES.items():
        if model_name in roc_data:
            data = roc_data[model_name]
            auroc = data.get("auroc", roc_auc_score(data["y_true"], data["y_score"]) if "y_score" in data else None)
            label = f"{style['label']} (AUROC = {auroc:.3f})" if auroc is not None else style['label']
            ax.plot(data["fpr"], data["tpr"],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], label=label)

    # Diagonal reference
    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0, label="Random Classifier")

    ax.set_xlabel("1 - Specificity (False Positive Rate)", fontsize=12)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=12)
    ax.set_title("A. Overall Temporal Validation Set (N = 2,000)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 2a saved to {save_path}")


def plot_roc_geriatric(
    roc_data_geriatric: Dict[str, Dict[str, np.ndarray]],
    save_path: str = "outputs/figures/figure2b_roc_geriatric.tiff",
    figsize: tuple = (8, 6),
    dpi: int = 300,
):
    """
    Plot ROC curves for the geriatric subgroup (age ≥ 65), Figure 2b.

    Args:
        roc_data_geriatric: Same structure as above, for the subgroup only.
        save_path: Save path.
        figsize: Figure size.
        dpi: Resolution.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Only show key models as per Figure 2B (DeepTriage-CN, TabNet, XGBoost, ESI, NEWS2)
    models_to_plot = ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)", "ESI", "NEWS2"]
    for model_name in models_to_plot:
        if model_name in roc_data_geriatric:
            data = roc_data_geriatric[model_name]
            style = MODEL_STYLES[model_name]
            auroc = data.get("auroc", None)
            label = f"{style['label']}"
            if auroc is not None:
                label += f" (AUROC = {auroc:.3f})"
            ax.plot(data["fpr"], data["tpr"],
                    color=style["color"], linestyle=style["linestyle"],
                    linewidth=style["linewidth"], label=label)

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.0, label="Random Classifier")

    ax.set_xlabel("1 - Specificity", fontsize=12)
    ax.set_ylabel("Sensitivity", fontsize=12)
    ax.set_title("B. Geriatric Subgroup (Age ≥ 65, n = 410)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 2b saved to {save_path}")