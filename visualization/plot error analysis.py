#!/usr/bin/env python3
"""
plot_error_analysis.py

Reproduces Figure 6: Error analysis charts.
    (a) Distribution of misclassifications (false positives/negatives).
    (b) False-negative rates stratified by age and narrative length.
"""

import numpy as np
import matplotlib.pyplot as plt
import os


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "outputs/figures/figure6a_error_distribution.tiff",
    figsize: tuple = (6, 5),
    dpi: int = 300,
):
    """
    Bar chart showing counts of True Positives, True Negatives, False Positives,
    False Negatives.
    """
    tp = np.sum((y_true == 1) & (y_pred == 1))
    tn = np.sum((y_true == 0) & (y_pred == 0))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))

    categories = ["True\nPositives", "True\nNegatives", "False\nPositives", "False\nNegatives"]
    values = [tp, tn, fp, fn]
    colors = ["#2ECC71", "#3498DB", "#E74C3C", "#E67E22"]

    fig, ax = plt.subplots(figsize=figsize)
    bars = ax.bar(categories, values, color=colors, edgecolor="black", linewidth=0.5)
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3, str(val),
                ha="center", va="bottom", fontsize=10)

    ax.set_ylabel("Count", fontsize=12)
    ax.set_title("A. Misclassification Distribution (Validation Set)", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 6a saved to {save_path}")


def plot_fnr_by_subgroup(
    fnr_data: dict,
    save_path: str = "outputs/figures/figure6b_fnr_subgroup.tiff",
    figsize: tuple = (8, 5),
    dpi: int = 300,
):
    """
    Error bar plot for false-negative rates stratified by age group and
    narrative length (sparse vs rich).

    Expected keys in fnr_data:
        "young_sparse", "elder_sparse", "young_rich", "elder_rich"
        each a tuple (fnr, n_admitted)
    """
    groups = ["Young (<65)\nSparse (≤3 words)", "Elder (≥65)\nSparse",
              "Young (<65)\nRich (>3 words)", "Elder (≥65)\nRich"]
    fnr_vals = []
    cis = []  # placeholder CI if available
    for key in ["young_sparse", "elder_sparse", "young_rich", "elder_rich"]:
        fnr, n = fnr_data.get(key, (0, 1))
        fnr_vals.append(fnr * 100)  # convert to percentage
        # Approximate 95% CI using binomial proportion
        if n > 0:
            p = fnr
            se = np.sqrt(p * (1 - p) / n)
            ci_low = max(0, p - 1.96 * se) * 100
            ci_high = min(1, p + 1.96 * se) * 100
            cis.append((ci_low, ci_high))
        else:
            cis.append((0, 0))

    yerr_low = [fnr_vals[i] - cis[i][0] for i in range(len(groups))]
    yerr_high = [cis[i][1] - fnr_vals[i] for i in range(len(groups))]

    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(len(groups))
    bars = ax.bar(x, fnr_vals, yerr=[yerr_low, yerr_high], capsize=5,
                  color=["#3498DB", "#E34A33", "#3498DB", "#E34A33"],
                  edgecolor="black", linewidth=0.5)

    for i, (bar, fnr) in enumerate(zip(bars, fnr_vals)):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f"{fnr:.1f}%", ha="center", fontsize=10)

    ax.set_xticks(x)
    ax.set_xticklabels(groups, fontsize=10)
    ax.set_ylabel("False Negative Rate (%)", fontsize=12)
    ax.set_title("B. False-Negative Rate by Age and Narrative Length", fontsize=13, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 6b saved to {save_path}")