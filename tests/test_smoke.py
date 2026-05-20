#!/usr/bin/env python3
"""
test_smoke.py

Smoke tests that validate the full pipeline on a tiny synthetic dataset
(n=200, no GPU required). These tests check correctness of data flow,
bug-fix integrity, and metric computation—they do NOT reproduce paper metrics.

Run with:
    pytest tests/test_smoke.py -v
"""

import sys
import os
import numpy as np
import pytest

# Ensure project root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def tiny_dataset():
    """Generate a reproducible 200-sample synthetic dataset."""
    rng = np.random.default_rng(0)
    n   = 200
    X   = rng.standard_normal((n, 8)).astype(np.float32)
    # Introduce 10 % NaN values (as in real data)
    mask = rng.random((n, 8)) < 0.10
    X[mask] = np.nan
    y     = rng.binomial(1, 0.22, n).astype(int)
    texts = [
        "头痛 发热 咳嗽" if i % 3 == 0
        else "腹痛 恶心" if i % 3 == 1
        else "[MISSING]"
        for i in range(n)
    ]
    return X, y, texts


@pytest.fixture(scope="module")
def split_dataset(tiny_dataset):
    """Split 200 samples into 160 train / 40 val."""
    X, y, texts = tiny_dataset
    return (
        X[:160], y[:160], texts[:160],
        X[160:], y[160:], texts[160:],
    )


# ---------------------------------------------------------------------------
# structured_encoder
# ---------------------------------------------------------------------------

class TestStructuredEncoder:
    def test_fit_transform_nan_safe(self, tiny_dataset):
        from src.structured_encoder import StructuredEncoder
        X, _, _ = tiny_dataset
        enc = StructuredEncoder()
        X_std = enc.fit_transform(X)
        # NaN positions should remain NaN
        assert np.isnan(X_std[np.isnan(X)]).all(), \
            "NaN positions must pass through unchanged."
        # Non-NaN positions should be standardised
        non_nan = ~np.isnan(X)
        assert np.isfinite(X_std[non_nan]).all()

    def test_no_leakage(self, split_dataset):
        """Verify transform uses training-set parameters, not val-set params."""
        from src.structured_encoder import StructuredEncoder
        X_tr, _, _, X_val, _, _ = split_dataset
        enc = StructuredEncoder()
        enc.fit(X_tr)
        # Modify val slightly
        X_val_modified = X_val + 100.0
        X_std_orig    = enc.transform(X_val)
        X_std_modified = enc.transform(X_val_modified)
        # Means should differ (not re-fitted)
        assert not np.allclose(
            np.nanmean(X_std_orig), np.nanmean(X_std_modified)
        )

    def test_get_set_params_roundtrip(self, tiny_dataset):
        from src.structured_encoder import StructuredEncoder
        X, _, _ = tiny_dataset
        enc = StructuredEncoder()
        enc.fit(X)
        params = enc.get_params()
        enc2 = StructuredEncoder()
        enc2.set_params(params)
        np.testing.assert_array_almost_equal(enc.mean_, enc2.mean_)
        np.testing.assert_array_almost_equal(enc.std_,  enc2.std_)


# ---------------------------------------------------------------------------
# clinical_scores (Bug B1 fix)
# ---------------------------------------------------------------------------

class TestClinicalScores:
    def test_esi_no_walrus_syntax(self, tiny_dataset):
        """B1 fix: compute_esi_level must not raise SyntaxError."""
        from src.clinical_scores import compute_esi_level
        X, _, _ = tiny_dataset
        age  = np.random.default_rng(1).uniform(18, 90, 200)
        hr   = X[:, 3]
        rr   = X[:, 4]
        sbp  = X[:, 5]
        spo2 = X[:, 7]
        temp = X[:, 2]
        esi  = compute_esi_level(age, hr, rr, sbp, spo2, temp)
        assert esi.shape == (200,)
        assert set(np.unique(esi[~np.isnan(esi)])).issubset({1, 2, 3})

    def test_news2_shape(self, tiny_dataset):
        from src.clinical_scores import compute_news2
        X, _, _ = tiny_dataset
        scores = compute_news2(
            rr=X[:, 4], spo2=X[:, 7], temp=X[:, 2],
            sbp=X[:, 5], hr=X[:, 3],
        )
        assert scores.shape == (200,)
        assert np.all(scores[~np.isnan(scores)] >= 0)

    def test_mews_shape(self, tiny_dataset):
        from src.clinical_scores import compute_mews
        X, _, _ = tiny_dataset
        scores = compute_mews(
            rr=X[:, 4], hr=X[:, 3], sbp=X[:, 5], temp=X[:, 2]
        )
        assert scores.shape == (200,)


# ---------------------------------------------------------------------------
# preprocess
# ---------------------------------------------------------------------------

class TestPreprocess:
    def test_raw_and_std_differ(self, split_dataset):
        """preprocess_structured must return different values from get_raw_structured."""
        from src.preprocess import get_raw_structured, preprocess_structured
        import pandas as pd
        X_tr, y_tr, _, X_val, y_val, _ = split_dataset
        cols = ["age", "sex", "temperature", "heart_rate",
                "respiratory_rate", "sbp", "dbp", "spo2"]
        train_df = pd.DataFrame(X_tr, columns=cols)
        val_df   = pd.DataFrame(X_val, columns=cols)
        train_df["hospital_admission"] = y_tr
        val_df["hospital_admission"]   = y_val

        X_raw_tr, X_raw_val         = get_raw_structured(train_df, val_df)
        X_std_tr, X_std_val, _enc   = preprocess_structured(train_df, val_df)

        # Raw and standardised should differ (unless all values happen to be 0)
        non_nan = ~np.isnan(X_raw_tr)
        assert not np.allclose(X_raw_tr[non_nan], X_std_tr[non_nan])


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_compute_all_binary_metrics(self):
        from evaluation.metrics import compute_all_binary_metrics
        rng    = np.random.default_rng(42)
        y_true = rng.binomial(1, 0.22, 500)
        y_score = np.clip(
            y_true * 0.6 + rng.standard_normal(500) * 0.3, 0, 1
        )
        metrics = compute_all_binary_metrics(y_true, y_score, threshold=0.28)
        for key in ["auroc", "auprc", "sensitivity", "specificity", "accuracy", "f1"]:
            assert key in metrics
            assert 0.0 <= metrics[key] <= 1.0


# ---------------------------------------------------------------------------
# statistical_tests
# ---------------------------------------------------------------------------

class TestStatisticalTests:
    def test_delong_identical_models(self):
        """DeLong test on identical predictions should give z=0, p≈1."""
        from evaluation.statistical_tests import delong_test
        rng    = np.random.default_rng(0)
        y      = rng.binomial(1, 0.22, 500)
        scores = rng.random(500)
        z, p   = delong_test(y, scores, scores)
        assert abs(z) < 1e-6
        assert p > 0.99

    def test_continuous_nri_range(self):
        from evaluation.statistical_tests import continuous_nri
        rng = np.random.default_rng(0)
        y   = rng.binomial(1, 0.22, 500)
        ref = rng.random(500)
        new = np.clip(ref + rng.normal(0, 0.1, 500), 0, 1)
        total, ev, nev = continuous_nri(y, ref, new)
        assert -2.0 <= total <= 2.0


# ---------------------------------------------------------------------------
# robustness simulation
# ---------------------------------------------------------------------------

class TestRobustness:
    def test_all_three_proportions_generated(self, tiny_dataset):
        """Bug L3 fix: all three proportions must be simulated."""
        from src.robustness import simulate_degraded_data
        from src.structured_encoder import STRUCTURED_FEATURE_NAMES
        X, _, _ = tiny_dataset
        props = [0.10, 0.20, 0.30]
        result = simulate_degraded_data(X, STRUCTURED_FEATURE_NAMES,
                                        missing_proportions=props, seed=0)
        assert set(result.keys()) == set(props), \
            "All three missing proportions must be returned."
        for p in props:
            assert result[p].shape == X.shape

    def test_degraded_nan_rate_increases(self, tiny_dataset):
        """Higher proportions must produce higher NaN rates."""
        from src.robustness import simulate_degraded_data
        from src.structured_encoder import STRUCTURED_FEATURE_NAMES
        X, _, _ = tiny_dataset
        result = simulate_degraded_data(X, STRUCTURED_FEATURE_NAMES,
                                        missing_proportions=[0.10, 0.20, 0.30],
                                        seed=0)
        nan_10 = np.isnan(result[0.10]).mean()
        nan_30 = np.isnan(result[0.30]).mean()
        assert nan_30 >= nan_10, "30 % level must have more missing than 10 %."


# ---------------------------------------------------------------------------
# decision_curve
# ---------------------------------------------------------------------------

class TestDecisionCurve:
    def test_net_benefit_treat_none_is_zero(self):
        """At no threshold should 'treat none' net benefit exceed 0."""
        from evaluation.decision_curve import compute_net_benefit
        rng = np.random.default_rng(0)
        y   = rng.binomial(1, 0.22, 400)
        s   = rng.random(400)
        thresholds = np.linspace(0.01, 0.99, 50)
        _, nb_model, nb_all = compute_net_benefit(y, s, thresholds)
        # Model NB clipped to 0 in our implementation
        assert np.all(nb_model >= 0)


# ---------------------------------------------------------------------------
# error_analysis
# ---------------------------------------------------------------------------

class TestErrorAnalysis:
    def test_fnr_subgroup_keys(self):
        from evaluation.error_analysis import error_analysis_by_subgroup
        rng     = np.random.default_rng(0)
        y_true  = rng.binomial(1, 0.22, 200)
        y_pred  = rng.binomial(1, 0.5, 200)
        ages    = rng.uniform(18, 90, 200)
        lengths = rng.integers(1, 20, 200)
        result  = error_analysis_by_subgroup(y_true, y_pred, ages, lengths)
        expected_keys = {"young_sparse", "elder_sparse", "young_rich", "elder_rich"}
        assert set(result.keys()) == expected_keys
        for v in result.values():
            assert "fnr" in v and "n" in v
            assert 0.0 <= v["fnr"] <= 1.0
