#!/usr/bin/env python3
"""
error_analysis.py

Error analysis by age × narrative-length subgroup (Figure 6 of the paper).

Paper Section 4.7:
    "False negatives … were disproportionately found among encounters with
    extremely brief chief complaints (≤3 words, e.g., 'dizziness')"
    "The false-negative rate reached 6.1% in older patients with sparse
    narratives, versus 3.2% in younger counterparts."
    "n=49 for age ≥65 years with sparse narratives, and n=125 for age <65
    with sparse narratives."

Subgroup definitions:
    Sparse narrative : chief complaint word count ≤ 3 (Section 4.7)
    Geriatric        : age ≥ 65 years (Section 3.2)
"""

import numpy as np
from typing import Dict, Any


def _fnr_for_group(
    y_true_group: np.ndarray,
    y_pred_group: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute false-negative rate and sample size for a subgroup.

    FNR = FN / (FN + TP) = FN / (total positives in group)

    Returns dict with keys 'fnr' (float) and 'n' (int).
    """
    n        = len(y_true_group)
    pos_mask = y_true_group == 1
    n_pos    = pos_mask.sum()

    if n_pos == 0:
        return {"fnr": 0.0, "n": n}

    fn  = np.sum((y_pred_group[pos_mask] == 0))
    fnr = fn / n_pos
    return {"fnr": float(fnr), "n": int(n)}


def error_analysis_by_subgroup(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    ages: np.ndarray,
    complaint_lengths: np.ndarray,
    age_threshold: int = 65,
    sparse_narrative_threshold: int = 3,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute FNR for four age × narrative-length subgroups.

    Subgroups:
        young_sparse : age < 65, word count ≤ sparse_narrative_threshold
        elder_sparse : age ≥ 65, word count ≤ sparse_narrative_threshold
        young_rich   : age < 65, word count >  sparse_narrative_threshold
        elder_rich   : age ≥ 65, word count >  sparse_narrative_threshold

    Args:
        y_true                     : Ground-truth labels.
        y_pred                     : Binary predictions.
        ages                       : Patient ages in years.
        complaint_lengths          : WORD counts of chief complaints
                                     (str.split().str.len(), Section 4.7).
        age_threshold              : Geriatric cutoff (default 65).
        sparse_narrative_threshold : Sparse narrative cutoff in words
                                     (paper definition: ≤ 3 words).

    Returns:
        Dict mapping subgroup name → {'fnr': float, 'n': int}.
    """
    y_true  = np.asarray(y_true,  dtype=int)
    y_pred  = np.asarray(y_pred,  dtype=int)
    ages    = np.asarray(ages,    dtype=float)
    lengths = np.asarray(complaint_lengths, dtype=float)

    elder  = ages >= age_threshold
    young  = ~elder
    sparse = lengths <= sparse_narrative_threshold
    rich   = ~sparse

    results: Dict[str, Dict[str, Any]] = {}
    for name, mask in [
        ("young_sparse", young & sparse),
        ("elder_sparse", elder & sparse),
        ("young_rich",   young & rich),
        ("elder_rich",   elder & rich),
    ]:
        results[name] = _fnr_for_group(y_true[mask], y_pred[mask])

    return results
