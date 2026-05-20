#!/usr/bin/env python3
"""
plot_decision_curve.py

Reproduces Figure 4: Decision Curve Analysis comparing DeepTriage-CN
and NEWS2 against 'treat all' and 'treat none' strategies.

Bug fixed (B2): The original file used a relative import:
    from ..evaluation.decision_curve import compute_net_benefit
However, 'visualization' and 'evaluation' are sibling packages under the
project root — neither is a sub-package of the other — so '..' ascends to
the parent of 'visualization', which is the project root, and 'evaluation'
does not exist there as a relative module.  This caused:
    ImportError: attempted relative import beyond top-level package

Fix: Use an absolute import, which works correctly when the project root is
on sys.path (ensured by all entry-point scripts).
"""

import numpy as np
import matplotlib.pyplot as plt
import os

from evaluation.decision_curve import compute_net_benefit   # ← absolute import


def plot_decision_curve(
    y_true: np.ndarray,
    deep_triage_proba: np.ndarray,
    news2_scores: np.ndarray = None,
    save_path: str = "outputs/figures/figure4_decision_curve.tiff",
    figsize: tuple = (8, 6),
    dpi: int = 300,
):
    """
    Plot Decision Curve Analysis for DeepTriage-CN vs NEWS2 and default strategies.

    Args:
        y_true            : Binary admission labels (0 or 1).
        deep_triage_proba : DeepTriage-CN predicted probabilities.
        news2_scores      : Raw NEWS2 integer scores (0–20).
                            If None, only DeepTriage-CN is plotted.
        save_path         : Output file path.
        figsize           : Matplotlib figure size.
        dpi               : Figure resolution.
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    _, nb_deep, nb_all = compute_net_benefit(y_true, deep_triage_proba, thresholds)

    fig, ax = plt.subplots(figsize=figsize)

    # DeepTriage-CN
    ax.plot(thresholds, nb_deep,
            color="#E34A33", linewidth=2, label="DeepTriage-CN (Multimodal)")

    # NEWS2 (normalised to [0, 1] probability scale)
    if news2_scores is not None:
        news2_prob = np.asarray(news2_scores, dtype=float) / 20.0
        _, nb_news2, _ = compute_net_benefit(y_true, news2_prob, thresholds)
        ax.plot(thresholds, nb_news2,
                color="#8E44AD", linestyle="--", linewidth=2, label="NEWS2")

    # Reference strategies
    ax.plot(thresholds, nb_all,
            color="gray", linestyle=":", linewidth=2, label="Treat All")
    ax.axhline(y=0, color="black", linestyle="-", linewidth=2, label="Treat None")

    ax.set_xlabel("Threshold Probability", fontsize=12)
    ax.set_ylabel("Net Benefit", fontsize=12)
    ax.set_title(
        "Decision Curve Analysis for Hospital Admission Prediction",
        fontsize=14, fontweight="bold",
    )
    ax.legend(loc="best", fontsize=10)
    ax.set_xlim([0.0, 1.0])

    # Set y-limits with a small upper margin
    upper = max(nb_all.max(), nb_deep.max()) + 0.02
    ax.set_ylim([-0.05, upper])
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 4 saved → {save_path}")
