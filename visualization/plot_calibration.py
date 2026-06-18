#!/usr/bin/env python3
"""
plot_calibration.py

Generates Figure 3: Calibration plot with density sub-panel.

Paper Figure 3 caption:
    "The solid vermilion line represents the observed calibration mapped
     across probability deciles, while the diagonal dotted grey line
     indicates the ideal perfect calibration. The model exhibits high
     reliability and closely follows the diagonal, yielding a Brier score
     of 0.12. The bottom sub-panel displays the density distribution of
     the model's predicted probabilities."

Key calibration numbers from paper (Section 4.3):
    Brier score = 0.12  |  slope = 0.95  |  intercept = 0.02
    Hosmer–Lemeshow p = 0.28
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from sklearn.calibration import calibration_curve


def plot_calibration_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    brier_score: float,
    n_bins: int = 10,
    save_path: str = "outputs/figures/figure3_calibration.tiff",
) -> None:
    """
    Plot calibration curve with probability density sub-panel (Figure 3).

    Args:
        y_true       : Ground-truth binary labels.
        y_score      : DeepTriage-CN predicted probabilities.
        brier_score  : Pre-computed Brier score for annotation.
        n_bins       : Number of calibration bins (default 10 = deciles).
        save_path    : Output TIFF path.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    fig = plt.figure(figsize=(4.5, 5.0))
    gs  = gridspec.GridSpec(2, 1, height_ratios=[3, 1], hspace=0.08)
    ax_cal = fig.add_subplot(gs[0])
    ax_den = fig.add_subplot(gs[1], sharex=ax_cal)

    # --- Calibration curve ---
    fraction_positive, mean_predicted = calibration_curve(
        y_true, y_score, n_bins=n_bins, strategy="quantile"
    )

    ax_cal.plot([0, 1], [0, 1], linestyle=":", color="0.5", linewidth=0.8,
                label="Perfect calibration")
    ax_cal.plot(mean_predicted, fraction_positive,
                color="#D55E00", marker="o", markersize=4, linewidth=1.8,
                label=f"DeepTriage-CN (Brier = {brier_score:.2f})")

    ax_cal.set_ylabel("Observed fraction admitted", fontsize=7)
    ax_cal.set_ylim([0, 1])
    ax_cal.legend(loc="upper left", fontsize=6)
    ax_cal.set_title("Calibration Plot — DeepTriage-CN", fontsize=8, fontweight="bold")
    plt.setp(ax_cal.get_xticklabels(), visible=False)

    # --- Density sub-panel ---
    ax_den.hist(y_score, bins=40, color="#D55E00", alpha=0.7, edgecolor="none")
    ax_den.set_xlabel("Predicted probability of admission", fontsize=7)
    ax_den.set_ylabel("Count", fontsize=7)
    ax_den.set_xlim([0, 1])

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 3 saved → {save_path}")
