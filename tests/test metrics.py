#!/usr/bin/env python3
"""
test_metrics.py

Unit tests for evaluation metrics: AUROC, AUPRC, calibration, bootstrap,
DeLong, NRI, and error analysis.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from evaluation.metrics import compute_all_binary_metrics, find_youden_threshold
from evaluation.calibration import (
    brier_score,
    calibration_intercept_slope,
    hosmer_lemeshow_test,
    compute_calibration_metrics,
)
from evaluation.bootstrap import bootstrap_confidence_intervals
from evaluation.statistical_tests import delong_test, continuous_nri
from evaluation.error_analysis import error_analysis_by_subgroup


@pytest.fixture
def perfect_preds():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.2, 0.1, 0.05, 0.9, 0.8, 0.95, 0.85])
    return y_true, y_proba


def test_compute_metrics_perfect(perfect_preds):
    y_true, y_proba = perfect_preds
    metrics = compute_all_binary_metrics(y_true, y_proba, threshold=0.5)
    assert metrics["auroc"] == 1.0
    assert metrics["sensitivity"] == 1.0
    assert metrics["specificity"] == 1.0


def test_youden_threshold():
    y_true = np.array([0, 0, 0, 0, 1, 1, 1, 1])
    y_proba = np.array([0.1, 0.3, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9])
    thresh = find_youden_threshold(y_true, y_proba)
    assert 0 < thresh < 1.0


def test_brier_score():
    y_true = np.array([0, 1])
    y_pred = np.array([0.0, 1.0])
    assert brier_score(y_true, y_pred) == 0.0
    y_pred = np.array([0.5, 0.5])
    assert brier_score(y_true, y_pred) == 0.25


def test_calibration_intercept_slope():
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0.2, 0.8, 0.2, 0.8])
    intercept, slope = calibration_intercept_slope(y_true, y_pred)
    # Should be approximately 0 and 1 for well-calibrated
    assert abs(intercept) < 1.0
    assert slope > 0


def test_hosmer_lemeshow():
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_pred = np.array([0.1, 0.9, 0.2, 0.8, 0.1, 0.9, 0.2, 0.8])
    _, p = hosmer_lemeshow_test(y_true, y_pred, n_bins=4)
    # Should not be extremely significant if well-calibrated
    assert p > 0.01


def test_bootstrap():
    y_true = np.random.binomial(1, 0.3, 100)
    y_score = y_true + np.random.normal(0, 0.1, 100)
    ci = bootstrap_confidence_intervals(y_true, y_score, n_bootstrap=100)
    assert "auroc" in ci
    point, low, high = ci["auroc"]
    assert 0 <= low <= point <= high <= 1.0


def test_delong():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y1 = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    y2 = np.array([0.1, 0.1, 0.2, 0.6, 0.7, 0.8])
    z, p = delong_test(y_true, y1, y2)
    assert isinstance(z, float)
    assert 0 <= p <= 1.0


def test_continuous_nri():
    y_true = np.array([0, 0, 1, 1, 0, 1])
    y_old = np.array([0.1, 0.4, 0.5, 0.6, 0.3, 0.7])
    y_new = np.array([0.2, 0.3, 0.6, 0.7, 0.2, 0.8])
    nri_total, nri_events, nri_nonevents = continuous_nri(y_true, y_old, y_new)
    assert isinstance(nri_total, float)
    assert -1.0 <= nri_total <= 1.0  # conservative bounds


def test_error_analysis():
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_pred = np.array([0, 0, 1, 1, 1, 0])  # one false positive, one false negative
    ages = np.array([70, 30, 50, 80, 25, 66])
    lens = np.array([3, 12, 4, 2, 15, 5])
    res = error_analysis_by_subgroup(y_true, y_pred, ages, lens,
                                     age_threshold=65, sparse_narrative_threshold=5)
    assert "overall" in res
    assert res["elder_sparse"]["fnr"] >= 0
    assert res["young_rich_fpr"]["fpr"] >= 0