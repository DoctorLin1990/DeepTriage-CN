#!/usr/bin/env python3
"""
plot_robustness.py

Generates Supplementary Figure 1: Robustness bar chart comparing baseline
vs. degraded AUROC under 30% MNAR missingness + Gaussian noise (Section 3.7).

Paper Supplementary Figure 1 caption:
    "Bars display the mean AUROC for the primary algorithms at baseline (full
     data, solid bars) and under a simulated missing-not-at-random (MNAR)
     condition with 30% vital sign omission and Gaussian noise injection
     (σ=0.5, hatched bars). The multimodal framework (DeepTriage-CN) retains
     95.4% of its baseline discriminative performance, degrading more
     gracefully than the structured-only deep learner (TabNet, 83.2% retained)
     and the vitals-only gradient boosting baseline (XGBoost, 82.8% retained)."

Key numbers aligned with paper (Section 4.5):
    DeepTriage-CN : baseline 0.865 → degraded 0.825 → retention 95.4%
    TabNet        : baseline 0.867 → degraded 0.722 → retention 83.2%
    XGBoost       : baseline 0.858 → degraded 0.710 → retention 82.8%
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Dict


COLORS = {
    "DeepTriage-CN":          "#D55E00",
    "TabNet":                 "#009E73",
    "XGBoost (Vitals-only)":  "#56B4E9",
}

MODEL_ORDER = ["DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)"]
SHORT_LABELS = {
    "DeepTriage-CN":          "DeepTriage-CN\n(Multimodal)",
    "TabNet":                 "TabNet\n(Structured DL)",
    "XGBoost (Vitals-only)":  "XGBoost\n(Vitals-only)",
}


def plot_robustness_bars(
    baseline_aucs: Dict[str, float],
    degraded_aucs: Dict[str, float],
    save_path: str = "outputs/figures/supp_figure1_robustness.tiff",
    figsize: tuple = (5.5, 4.0),
    dpi: int = 300,
) -> None:
    """
    Bar chart comparing baseline and degraded AUROC for each model.

    Args:
        baseline_aucs : Dict mapping model name → baseline AUROC.
        degraded_aucs : Dict mapping model name → AUROC under 30% MNAR + noise.
        save_path     : Output TIFF path.
        figsize       : Figure dimensions in inches.
        dpi           : Figure resolution.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
        "axes.linewidth": 0.8,
        "xtick.direction": "in",
        "ytick.direction": "in",
    })

    models   = [m for m in MODEL_ORDER if m in baseline_aucs]
    x        = np.arange(len(models))
    bar_w    = 0.35

    fig, ax = plt.subplots(figsize=figsize)

    # Baseline bars (solid)
    bars_base = ax.bar(
        x - bar_w / 2,
        [baseline_aucs[m] for m in models],
        bar_w,
        color=[COLORS[m] for m in models],
        alpha=0.85,
        label="Baseline (Full Data)",
        edgecolor="white",
        linewidth=0.4,
    )

    # Degraded bars (hatched)
    bars_deg = ax.bar(
        x + bar_w / 2,
        [degraded_aucs[m] for m in models],
        bar_w,
        color=[COLORS[m] for m in models],
        alpha=0.55,
        hatch="//",
        label="Degraded (30% MNAR + Noise, σ=0.5)",
        edgecolor="white",
        linewidth=0.4,
    )

    # Retention annotation
    for i, m in enumerate(models):
        b  = baseline_aucs[m]
        d  = degraded_aucs[m]
        rt = d / b * 100
        ax.text(
            x[i] + bar_w / 2, d + 0.005,
            f"{rt:.1f}%",
            ha="center", va="bottom", fontsize=6, color="0.3",
        )

    ax.set_ylabel("AUROC", fontsize=8)
    ax.set_ylim([0.60, 0.94])
    ax.set_xticks(x)
    ax.set_xticklabels([SHORT_LABELS[m] for m in models], fontsize=7)
    ax.legend(loc="lower right", fontsize=6)
    ax.set_title(
        "Robustness Under Simulated Data Degradation (30% MNAR + Gaussian Noise)",
        fontsize=7, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    # Caution note
    ax.text(
        0.5, -0.18,
        "Note: Differential retention partially reflects structured simulation parameters "
        "(see Section 4.5).",
        transform=ax.transAxes, ha="center", fontsize=5, color="0.5",
        style="italic",
    )

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Supplementary Figure 1 saved → {save_path}")
