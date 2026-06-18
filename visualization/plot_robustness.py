#!/usr/bin/env python3
"""
visualization/plot_robustness.py
=================================
Supplementary Figure 1: AUROC retention under MNAR data degradation.
(Section 3.7 / Section 4.5)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

FONT = "DejaVu Sans"
PAL  = {
    "DeepTriage-CN":         "#D55E00",
    "TabNet":                "#009E73",
    "XGBoost (Vitals-only)": "#56B4E9",
}
MARKERS = {
    "DeepTriage-CN":         "o",
    "TabNet":                "s",
    "XGBoost (Vitals-only)": "^",
}
# Paper-reported retention values (Section 4.5)
PAPER_RETENTION = {
    "DeepTriage-CN":         {10: None, 20: None, 30: 0.954},
    "TabNet":                {10: None, 20: None, 30: 0.833},
    "XGBoost (Vitals-only)": {10: None, 20: None, 30: 0.828},
}


def _style():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": [FONT, "Arial"],
        "font.size": 7, "axes.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.major.width": 0.8, "ytick.major.width": 0.8,
        "xtick.major.size": 3.5, "ytick.major.size": 3.5,
        "xtick.direction": "in", "ytick.direction": "in",
        "legend.fontsize": 5.8, "legend.frameon": False,
        "figure.dpi": 300, "savefig.dpi": 300, "pdf.fonttype": 42,
    })


def plot_robustness(
    df: pd.DataFrame,
    out_path: str = "outputs/figures/SuppFigure1_Robustness.tiff",
) -> None:
    """
    Parameters
    ----------
    df : pd.DataFrame
        Must have columns: model, missing_proportion, auroc, auroc_baseline,
        auroc_retention (= auroc / auroc_baseline).
        One row per (model, missing_proportion, repetition).
    out_path : str
        Path for the output TIFF file.
    """
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.40))
    fig.subplots_adjust(wspace=0.38)

    models = list(PAL.keys())
    # Aggregate by model × missing_proportion
    if "auroc_retention" not in df.columns:
        df["auroc_retention"] = df["auroc"] / df["auroc_baseline"]
    agg = (df.groupby(["model", "missing_proportion"])
             .agg(auroc_mean=("auroc", "mean"),
                  auroc_std=("auroc",  "std"),
                  ret_mean=("auroc_retention", "mean"),
                  ret_std=("auroc_retention",  "std"))
             .reset_index())

    proportions = sorted(df["missing_proportion"].unique())

    # ── Left: absolute AUROC ────────────────────────────────────────────
    ax0 = axes[0]
    for model in models:
        sub = agg[agg["model"] == model].sort_values("missing_proportion")
        if len(sub) == 0:
            continue
        ax0.errorbar(
            sub["missing_proportion"] * 100,
            sub["auroc_mean"],
            yerr=sub["auroc_std"],
            color=PAL[model], marker=MARKERS[model],
            markersize=4.5, lw=1.5, capsize=3, capthick=0.8,
            label=model,
        )
    ax0.set_xlabel("Missing proportion (%)", fontsize=7)
    ax0.set_ylabel("AUROC", fontsize=7)
    ax0.set_xticks([p*100 for p in proportions])
    ax0.set_ylim([0.65, 0.92])
    ax0.legend(loc="lower left", fontsize=5.8)
    ax0.text(-0.13, 1.05, "a", transform=ax0.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="left")
    ax0.spines["top"].set_visible(False); ax0.spines["right"].set_visible(False)

    # ── Right: retention (%) ────────────────────────────────────────────
    ax1 = axes[1]
    for model in models:
        sub = agg[agg["model"] == model].sort_values("missing_proportion")
        if len(sub) == 0:
            continue
        ax1.errorbar(
            sub["missing_proportion"] * 100,
            sub["ret_mean"]  * 100,
            yerr=sub["ret_std"] * 100,
            color=PAL[model], marker=MARKERS[model],
            markersize=4.5, lw=1.5, capsize=3, capthick=0.8,
            label=model,
        )
        # Annotate the 30% missingness retention value from the paper
        ret_30 = PAPER_RETENTION.get(model, {}).get(30)
        if ret_30 is not None:
            ax1.annotate(
                f"{ret_30*100:.1f}%",
                xy=(30, ret_30 * 100),
                xytext=(32, ret_30 * 100 + 0.4),
                fontsize=5.2, color=PAL[model],
                arrowprops=dict(arrowstyle="-", color=PAL[model], lw=0.5),
            )

    ax1.set_xlabel("Missing proportion (%)", fontsize=7)
    ax1.set_ylabel("AUROC retention (%)", fontsize=7)
    ax1.set_xticks([p*100 for p in proportions])
    ax1.set_ylim([78, 102])
    ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0))
    ax1.axhline(100, color="0.70", lw=0.6, ls="--")
    ax1.legend(loc="lower left", fontsize=5.8)
    ax1.text(-0.13, 1.05, "b", transform=ax1.transAxes,
             fontsize=9, fontweight="bold", va="top", ha="left")
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    # Caption note (inside figure, bottom)
    fig.text(0.50, 0.00,
             "Note: Robustness simulation assumes MNAR near healthy baselines "
             "(Section 3.7). High-acuity missingness not captured. "
             "Results are provisional.",
             ha="center", va="bottom", fontsize=5.0, color="0.50",
             wrap=True)

    fig.tight_layout(pad=0.4, rect=[0, 0.06, 1, 1])
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=300, format="tiff", pad_inches=0.05,
                pil_kwargs={"compression": "tiff_lzw"})
    print(f"  ✓  {out_path}")
    plt.close(fig)
