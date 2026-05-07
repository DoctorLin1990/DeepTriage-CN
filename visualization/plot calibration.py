#!/usr/bin/env python3
"""
plot_calibration.py

Reproduces Figure 3: Calibration plot with density sub-panel.
Shows model-predicted probabilities vs. observed admission proportions,
along with histogram of predicted probabilities.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from typing import Optional
import os


def plot_calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    brier_score: Optional[float] = None,
    save_path: str = "outputs/figures/figure3_calibration.tiff",
    figsize: tuple = (7, 8),
    dpi: int = 300,
):
    """
    Plot calibration curve with density sub-panel, matching Figure 3.

    Args:
        y_true: True binary labels.
        y_pred: Predicted probabilities from DeepTriage-CN.
        n_bins: Number of bins for calibration curve.
        brier_score: Precomputed Brier score to display.
        save_path: Path to save figure.
        figsize: Figure size.
        dpi: Resolution.
    """
    # Create figure with two subplots: main calibration plot and density below
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[4, 1], hspace=0.25)
    ax_cal = fig.add_subplot(gs[0])
    ax_hist = fig.add_subplot(gs[1])

    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=n_bins)
    ax_cal.plot(prob_pred, prob_true, marker="o", color="#E34A33", linewidth=2, markersize=8, label="DeepTriage-CN")
    ax_cal.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1.5, label="Perfect Calibration")

    # Annotate Brier score
    if brier_score is not None:
        ax_cal.text(0.05, 0.95, f"Brier Score = {brier_score:.2f}", transform=ax_cal.transAxes,
                    fontsize=12, verticalalignment="top", bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    ax_cal.set_xlabel("Predicted Probability", fontsize=12)
    ax_cal.set_ylabel("Observed Admission Proportion", fontsize=12)
    ax_cal.set_title("Calibration of DeepTriage-CN", fontsize=14, fontweight="bold")
    ax_cal.legend(loc="lower right")
    ax_cal.set_xlim([0.0, 1.0])
    ax_cal.set_ylim([0.0, 1.0])
    ax_cal.grid(alpha=0.3)

    # Density histogram
    ax_hist.hist(y_pred, bins=50, color="#E34A33", alpha=0.7, edgecolor="black", linewidth=0.5)
    ax_hist.set_xlabel("Predicted Probability", fontsize=12)
    ax_hist.set_ylabel("Frequency", fontsize=10)
    ax_hist.set_xlim([0.0, 1.0])
    ax_hist.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 3 saved to {save_path}")