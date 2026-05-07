#!/usr/bin/env python3
"""
plot_decision_curve.py

Reproduces Figure 4: Decision Curve Analysis comparing DeepTriage-CN
and NEWS2 against 'treat all' and 'treat none' strategies.
"""

import numpy as np
import matplotlib.pyplot as plt
from ..evaluation.decision_curve import compute_net_benefit
import os


def plot_decision_curve(
    y_true: np.ndarray,
    deep_triage_proba: np.ndarray,
    news2_scores: np.ndarray = None,
    save_path: str = "outputs/figures/figure4_decision_curve.tiff",
    figsize: tuple = (8, 6),
    dpi: int = 300,
):
    """
    Plot decision curve for DeepTriage-CN vs NEWS2 and default strategies.

    Args:
        y_true: Binary labels.
        deep_triage_proba: DeepTriage-CN predicted probabilities.
        news2_scores: NEWS2 integer scores (0-20). If None, only DeepTriage-CN shown.
        save_path: Save path.
        figsize: Figure size.
        dpi: Resolution.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    _, nb_deep, nb_all = compute_net_benefit(y_true, deep_triage_proba, thresholds)

    fig, ax = plt.subplots(figsize=figsize)

    # DeepTriage-CN
    ax.plot(thresholds, nb_deep, color="#E34A33", linewidth=2, label="DeepTriage-CN (Multimodal)")

    # NEWS2 (if provided, scale scores to probabilities empirically or use raw scores as probabilities)
    if news2_scores is not None:
        # Normalize NEWS2 to [0,1] roughly (max possible score ~20)
        news2_prob = news2_scores / 20.0
        _, nb_news2, _ = compute_net_benefit(y_true, news2_prob, thresholds)
        ax.plot(thresholds, nb_news2, color="#8E44AD", linestyle="--", linewidth=2, label="NEWS2")

    # Treat all
    ax.plot(thresholds, nb_all, color="gray", linestyle=":", linewidth=2, label="Treat All")
    # Treat none (horizontal line at y=0)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=2, label="Treat None")

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title("Decision Curve Analysis for Hospital Admission Prediction", fontsize=14, fontweight="bold")
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([-0.05, max(nb_all.max(), nb_deep.max()) + 0.02])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 4 saved to {save_path}")