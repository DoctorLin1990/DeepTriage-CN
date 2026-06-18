#!/usr/bin/env python3
"""
plot_roc.py

Generates Figure 2 of the paper: ROC curves for overall (2a) and geriatric
subgroup (2b) validation sets.

Figure 2 caption (paper):
    "A. Predictive performance in the overall temporal validation set (N=2,000).
     DeepTriage-CN and the optimized TabNet exhibit comparable, leading
     discrimination.
     B. Predictive performance in the geriatric subgroup (age ≥65 years, n=410)."

Colour scheme (from 图片生成__2_.doc):
    DeepTriage-CN      : '#D55E00'  (vermilion / deep orange — core model)
    TabNet             : '#009E73'  (bluish green)
    XGBoost (Vitals)   : '#56B4E9'  (sky blue)
    Random Forest      : '#CC79A7'  (reddish purple)
    Text-Only (BERT+LR): '#F0E442'  (yellow)
    NEWS2              : '#E69F00'  (orange)  — dashed
    MEWS               : '#999999'  (grey)    — dashed
    ESI                : '#0072B2'  (blue)    — dashed

Style: Nature / Scientific Reports — Arial font, 7 pt, inward ticks,
       300 DPI TIFF, double-column width (7.1 in × 3.5 in per panel pair).
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Dict, Any


# ── Nature-style global settings ───────────────────────────────────────────
def _apply_nature_style() -> None:
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size":          7,
        "axes.linewidth":     0.8,
        "xtick.major.width":  0.8,
        "ytick.major.width":  0.8,
        "xtick.direction":   "in",
        "ytick.direction":   "in",
        "legend.fontsize":    6,
        "legend.frameon":     False,
    })


COLORS = {
    "DeepTriage-CN":          "#D55E00",
    "TabNet":                 "#009E73",
    "XGBoost (Vitals-only)":  "#56B4E9",
    "Random Forest":          "#CC79A7",
    "Text-Only (BERT+LR)":    "#F0E442",
    "NEWS2":                  "#E69F00",
    "MEWS":                   "#999999",
    "ESI":                    "#0072B2",
}

ML_MODELS   = {"DeepTriage-CN", "TabNet", "XGBoost (Vitals-only)",
               "Random Forest", "Text-Only (BERT+LR)"}
CLIN_MODELS = {"NEWS2", "MEWS", "ESI"}


def _plot_roc_on_ax(
    ax: plt.Axes,
    roc_data: Dict[str, Any],
    title: str,
    subtitle: str,
) -> None:
    """
    Draw ROC curves on a given Axes object.

    Args:
        ax       : Matplotlib Axes.
        roc_data : Dict mapping model name → {'fpr', 'tpr', 'auroc'}.
        title    : Panel title (e.g., 'a').
        subtitle : Panel subtitle text.
    """
    # Diagonal reference line
    ax.plot([0, 1], [0, 1], linestyle=":", color="0.6", linewidth=0.8,
            label="Random (AUROC = 0.50)")

    for name, data in roc_data.items():
        color     = COLORS.get(name, "#333333")
        linestyle = "--" if name in CLIN_MODELS else "-"
        lw        = 2.0 if name == "DeepTriage-CN" else 1.2
        ax.plot(
            data["fpr"], data["tpr"],
            color=color, linestyle=linestyle, linewidth=lw,
            label=f"{name} (AUROC = {data['auroc']:.3f})",
        )

    ax.set_xlabel("1 − Specificity (False Positive Rate)", fontsize=7)
    ax.set_ylabel("Sensitivity (True Positive Rate)", fontsize=7)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.text(-0.12, 1.04, title, transform=ax.transAxes,
            fontsize=9, fontweight="bold", va="top")
    ax.set_title(subtitle, fontsize=7, pad=4)
    ax.legend(loc="lower right", fontsize=5.5)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def plot_roc_overall(
    roc_data: Dict[str, Any],
    save_path: str = "outputs/figures/figure2a_roc_overall.tiff",
) -> None:
    """
    Plot Figure 2a: overall ROC curve for all models.

    Args:
        roc_data  : Dict mapping model name → {'fpr', 'tpr', 'auroc'}.
        save_path : Output TIFF path.
    """
    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _plot_roc_on_ax(
        ax, roc_data,
        title="a",
        subtitle="Overall validation set (N=2,000)",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 2a saved → {save_path}")


def plot_roc_geriatric(
    roc_data: Dict[str, Any],
    save_path: str = "outputs/figures/figure2b_roc_geriatric.tiff",
) -> None:
    """
    Plot Figure 2b: geriatric subgroup (age ≥65) ROC curve.

    Args:
        roc_data  : Dict mapping model name → {'fpr', 'tpr', 'auroc'}.
                    Should contain only models shown in the paper subgroup
                    analysis: DeepTriage-CN, TabNet, XGBoost (Vitals-only),
                    NEWS2, ESI.
        save_path : Output TIFF path.
    """
    _apply_nature_style()
    fig, ax = plt.subplots(figsize=(3.5, 3.5))
    _plot_roc_on_ax(
        ax, roc_data,
        title="b",
        subtitle="Geriatric subgroup (age ≥65 years, n=410)",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=300, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 2b saved → {save_path}")
