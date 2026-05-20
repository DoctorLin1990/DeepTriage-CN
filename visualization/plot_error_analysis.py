#!/usr/bin/env python3
"""
plot_error_analysis.py

Generates Figure 6 of the paper: error analysis at the Youden-optimal threshold.

Figure 6 caption (paper Section 4.7):
    a. Distribution of misclassifications in the temporal validation set.
       At the optimized threshold (0.28), false positives account for 14.8%
       of predictions, while false negatives constitute 2.7%.
    b. False-negative rates (FNR) stratified by age group and narrative length.
       Peak FNR 6.1% in geriatric patients with sparse narratives (≤3 words);
       3.2% in younger counterparts with equally sparse narratives.
       n=49 for age ≥65 with sparse narratives; n=125 for age <65.

Bug B5 fix (from AUDIT_REPORT.md):
    The original script passed plain floats as fnr_data values.
    plot_fnr_by_subgroup expects (fnr_float, n_int) tuples so that it can
    compute Wilson confidence intervals. generate_all_figures.py now correctly
    passes tuples; this function unpacks them.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple


def _wilson_ci(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    """
    Wilson score confidence interval for a proportion k/n.

    Returns (lower, upper) bounds in [0, 1].
    """
    if n == 0:
        return 0.0, 0.0
    p    = k / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    margin = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return max(0.0, center - margin), min(1.0, center + margin)


def plot_error_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "outputs/figures/figure6a_error_distribution.tiff",
    dpi: int = 300,
) -> None:
    """
    Plot Figure 6a: pie / bar chart of correct predictions and misclassifications.

    Paper numbers: FP = 14.8%, FN = 2.7%, correct = 82.5%.

    Args:
        y_true    : Ground-truth binary labels.
        y_pred    : Binary predictions at the Youden-optimal threshold.
        save_path : Output TIFF path.
        dpi       : Resolution.
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
    })

    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    n      = len(y_true)

    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))

    labels = ["True Positive\n(Correctly Admitted)",
              "True Negative\n(Correctly Discharged)",
              "False Positive\n(Over-predicted)",
              "False Negative\n(Missed Admission)"]
    sizes  = [tp, tn, fp, fn]
    colors = ["#56B4E9", "#009E73", "#F0E442", "#D55E00"]
    explode = [0, 0, 0.05, 0.10]

    fig, ax = plt.subplots(figsize=(5, 4))
    wedges, texts, autotexts = ax.pie(
        sizes, labels=labels, colors=colors, explode=explode,
        autopct=lambda p: f"{p:.1f}%\n(n={int(round(p*n/100))})",
        startangle=90, textprops={"fontsize": 6},
    )
    for at in autotexts:
        at.set_fontsize(5.5)

    ax.set_title(
        f"a   Classification Outcomes at Threshold 0.28\n"
        f"(N={n} encounters; admission rate {y_true.mean():.1%})",
        fontsize=7, fontweight="bold",
    )
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 6a saved → {save_path}")


def plot_fnr_by_subgroup(
    fnr_data: Dict[str, Tuple[float, int]],
    save_path: str = "outputs/figures/figure6b_fnr_subgroup.tiff",
    dpi: int = 300,
) -> None:
    """
    Plot Figure 6b: FNR by age × narrative-length subgroup with 95% CIs.

    Args:
        fnr_data  : Dict mapping subgroup name → (fnr_float, n_int) tuples.
                    Keys: 'young_sparse', 'elder_sparse', 'young_rich',
                           'elder_rich'.
                    n_int is the TOTAL subgroup size (denominator for CI).
        save_path : Output TIFF path.
        dpi       : Resolution.

    IMPORTANT (Bug B5 fix): values must be (fnr, n) tuples, NOT plain floats.
    generate_all_figures.py constructs them as:
        fnr_data = {
            "young_sparse": (err_res["young_sparse"]["fnr"],
                             err_res["young_sparse"]["n"]),
            ...
        }
    """
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7,
    })

    subgroup_labels = {
        "young_sparse": "Age <65\nSparse (≤3 words)",
        "elder_sparse": "Age ≥65\nSparse (≤3 words)",
        "young_rich":   "Age <65\nRich (>3 words)",
        "elder_rich":   "Age ≥65\nRich (>3 words)",
    }
    bar_colors = ["#56B4E9", "#D55E00", "#009E73", "#E69F00"]

    keys   = ["young_sparse", "elder_sparse", "young_rich", "elder_rich"]
    fnrs   = []
    lowers = []
    uppers = []
    ns     = []

    for key in keys:
        val = fnr_data.get(key, (0.0, 0))
        # Unpack (fnr_float, n_int) tuple — Bug B5 fix
        fnr, n = val
        fnrs.append(fnr)
        ns.append(n)
        k        = int(round(fnr * n))   # estimated FN count for CI
        lo, hi   = _wilson_ci(k, n)
        lowers.append(fnr - lo)
        uppers.append(hi - fnr)

    x       = np.arange(len(keys))
    bar_w   = 0.55

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    bars = ax.bar(
        x, fnrs, bar_w,
        color=bar_colors, alpha=0.85, edgecolor="white", linewidth=0.4,
    )
    ax.errorbar(
        x, fnrs,
        yerr=[lowers, uppers],
        fmt="none", color="0.2", capsize=4, linewidth=1.0,
    )

    # Annotate n
    for i, (fnr, n_val) in enumerate(zip(fnrs, ns)):
        ax.text(
            x[i], fnr + uppers[i] + 0.005,
            f"n={n_val}", ha="center", va="bottom", fontsize=5.5, color="0.4",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [subgroup_labels[k] for k in keys], fontsize=6.5,
    )
    ax.set_ylabel("False-Negative Rate (FNR)", fontsize=7)
    ax.set_ylim([0, max(fnrs) + max(uppers) + 0.04])
    ax.set_title(
        "b   False-Negative Rate by Age Group × Narrative Length\n"
        "(Error bars: 95% Wilson CI)",
        fontsize=7, fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3, linewidth=0.5)
    ax.set_axisbelow(True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
    fig.savefig(save_path, dpi=dpi, bbox_inches="tight", format="tiff")
    plt.close(fig)
    print(f"Figure 6b saved → {save_path}")
