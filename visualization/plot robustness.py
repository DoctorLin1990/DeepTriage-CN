#!/usr/bin/env python3
"""
plot_robustness.py

Reproduces Supplementary Figure 1: Model robustness under compounded data degradation.
Shows AUROC retention under 30% MNAR missingness + noise for DeepTriage-CN,
TabNet, and Vitals-only XGBoost.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_robustness_bars(
    auroc_baseline: dict,
    auroc_degraded: dict,
    labels: list = None,
    save_path: str = "outputs/figures/supp_figure1_robustness.tiff",
    figsize: tuple = (6, 5),
    dpi: int = 300,
):
    """
    Plot grouped bar chart comparing baseline vs degraded AUROC for each model.

    Args:
        auroc_baseline: Dictionary mapping model name -> baseline AUROC.
        auroc_degraded: Dictionary mapping model name -> degraded AUROC.
        labels: Ordered model names for x-axis. Default: DeepTriage-CN, TabNet, XGBoost.
        save_path: Save path.
        figsize: Figure size.
        dpi: Resolution.
    """
    if labels is None:
        labels = ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)"]

    baseline_vals = [auroc_baseline[name] for name in labels]
    degraded_vals = [auroc_degraded[name] for name in labels]
    retention = [degraded_vals[i] / baseline_vals[i] * 100 for i in range(len(labels))]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    bars1 = ax.bar(x - width/2, baseline_vals, width, label="Full Data (Baseline)",
                   color="#3498DB", edgecolor="black", linewidth=0.5)
    bars2 = ax.bar(x + width/2, degraded_vals, width, label="30% MNAR + Noise",
                   color="#E34A33", edgecolor="black", linewidth=0.5, hatch="//")

    # Annotate retention percentages
    for i, (b, d, r) in enumerate(zip(bars1, bars2, retention)):
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.005, f"{baseline_vals[i]:.3f}",
                ha="center", va="bottom", fontsize=9)
        ax.text(d.get_x() + d.get_width()/2, d.get_height() + 0.005, f"{degraded_vals[i]:.3f}\n({r:.1f}%)",
                ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("AUROC", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_title("Model Robustness Under Data Degradation", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim([0.65, max(baseline_vals) + 0.05])
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Supplementary Figure 1 saved to {save_path}")