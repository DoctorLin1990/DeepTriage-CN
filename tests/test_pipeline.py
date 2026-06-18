#!/usr/bin/env python3
"""
tests/test_pipeline.py
=======================
Unit and integration tests for the DeepTriage-CN pipeline.

Run with:  python -m pytest tests/test_pipeline.py -v
"""
import sys, warnings
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(42)


@pytest.fixture(scope="module")
def small_data(rng):
    """Minimal dataset: 200 samples, 8 struct features, 768 text dims."""
    N = 200
    y = (rng.random(N) < 0.22).astype(int)
    X_struct = np.column_stack([
        rng.integers(18, 90, N).astype(float),    # age
        rng.integers(0, 2, N).astype(float),       # sex
        rng.normal(37.0, 0.8, N),                  # temperature
        rng.normal(85.0, 15.0, N),                 # heart_rate
        rng.normal(18.0, 4.0, N),                  # respiratory_rate
        rng.normal(120.0, 20.0, N),                # sbp
        rng.normal(80.0,  10.0, N),                # dbp
        rng.normal(97.0,  3.0, N),                 # spo2
    ])
    # Introduce 5-10% missingness in vitals
    for j in range(2, 8):
        mask = rng.random(N) < 0.07
        X_struct[mask, j] = np.nan
    texts = ["发热 咳嗽"] * (N // 2) + ["胸痛 气促 大汗"] * (N // 2)
    return y, X_struct, texts


# ─────────────────────────────────────────────────────────────────────
# StructuredEncoder
# ─────────────────────────────────────────────────────────────────────

class TestStructuredEncoder:
    def test_fit_transform_shape(self, small_data):
        from src.structured_encoder import StructuredEncoder
        y, X, _ = small_data
        enc = StructuredEncoder()
        X_std = enc.fit_transform(X)
        assert X_std.shape == X.shape

    def test_nan_preserved(self, small_data):
        from src.structured_encoder import StructuredEncoder
        y, X, _ = small_data
        enc = StructuredEncoder()
        X_std = enc.fit_transform(X)
        # NaN positions must be preserved
        assert np.isnan(X_std).sum() == np.isnan(X).sum()

    def test_mean_near_zero(self, small_data):
        from src.structured_encoder import StructuredEncoder
        y, X, _ = small_data
        enc = StructuredEncoder()
        X_std = enc.fit_transform(X)
        col_means = np.nanmean(X_std, axis=0)
        assert np.allclose(col_means, 0, atol=0.05), \
            f"Column means not near zero: {col_means}"

    def test_transform_no_refit(self, small_data, rng):
        """transform() must use training stats, not refit on new data."""
        from src.structured_encoder import StructuredEncoder
        y, X, _ = small_data
        enc = StructuredEncoder()
        enc.fit_transform(X)
        # New data with different distribution
        X_new = X + 10.0
        X_new_std = enc.transform(X_new)
        # Should NOT be zero-centred (different from training distribution)
        col_means = np.nanmean(X_new_std, axis=0)
        assert not np.allclose(col_means, 0, atol=0.01), \
            "transform() appears to be refitting on new data (data leakage)."

    def test_constant_column_safe(self, rng):
        """Constant column (e.g., binary sex) must not cause divide-by-zero."""
        from src.structured_encoder import StructuredEncoder
        X = rng.normal(0, 1, (50, 3))
        X[:, 1] = 1.0   # constant column
        enc = StructuredEncoder()
        X_std = enc.fit_transform(X)
        assert np.isfinite(X_std).all()

    def test_serialisation(self, small_data, tmp_path):
        from src.structured_encoder import StructuredEncoder
        import joblib
        y, X, _ = small_data
        enc = StructuredEncoder()
        enc.fit_transform(X)
        path = tmp_path / "enc.pkl"
        joblib.dump(enc.get_params(), str(path))
        params = joblib.load(str(path))
        enc2 = StructuredEncoder()
        enc2.set_params(**params)
        X_std1 = enc.transform(X)
        X_std2 = enc2.transform(X)
        np.testing.assert_allclose(X_std1, X_std2, atol=1e-9)


# ─────────────────────────────────────────────────────────────────────
# TextEncoder
# ─────────────────────────────────────────────────────────────────────

class TestTextEncoder:
    def test_encode_shape(self):
        from src.text_encoder import TextEncoder
        te = TextEncoder()
        texts = ["发热 咳嗽", "胸痛 气促"]
        emb = te.encode(texts)
        assert emb.shape == (2, 768), f"Expected (2, 768), got {emb.shape}"

    def test_missing_token(self):
        from src.text_encoder import TextEncoder
        te = TextEncoder()
        emb_miss  = te.encode(["[MISSING]"])
        emb_empty = te.encode([""])
        # Both should produce finite embeddings
        assert np.isfinite(emb_miss).all()
        assert np.isfinite(emb_empty).all()

    def test_frozen_weights(self):
        """BERT weights must not change between two encode() calls."""
        from src.text_encoder import TextEncoder
        import torch
        te   = TextEncoder()
        text = ["胸痛 气促"]
        e1   = te.encode(text)
        e2   = te.encode(text)
        np.testing.assert_array_equal(e1, e2)

    def test_deterministic_batch(self):
        from src.text_encoder import TextEncoder
        te = TextEncoder()
        texts = ["发热"] * 10 + ["胸痛"] * 10
        e1 = te.encode(texts, batch_size=4)
        e2 = te.encode(texts, batch_size=8)
        np.testing.assert_allclose(e1, e2, atol=1e-5)


# ─────────────────────────────────────────────────────────────────────
# Clinical Scores
# ─────────────────────────────────────────────────────────────────────

class TestClinicalScores:
    def test_news2_range(self, small_data):
        from src.clinical_scores import compute_news2
        _, X, _ = small_data
        rr, spo2, temp, sbp, hr = X[:,4], X[:,7], X[:,2], X[:,5], X[:,3]
        scores = compute_news2(rr, spo2, temp, sbp, hr)
        assert ((scores >= 0) | np.isnan(scores)).all()
        assert ((scores <= 20) | np.isnan(scores)).all()

    def test_news2_nan_safe(self):
        from src.clinical_scores import compute_news2
        rr   = np.array([18.0, np.nan])
        spo2 = np.array([97.0, 95.0])
        temp = np.array([36.8, 37.1])
        sbp  = np.array([120.0, 110.0])
        hr   = np.array([75.0, 80.0])
        scores = compute_news2(rr, spo2, temp, sbp, hr)
        assert np.isfinite(scores[0])   # First should compute fine
        # Second (NaN rr) should produce NaN or valid partial score
        assert scores.shape == (2,)

    def test_mews_non_negative(self, small_data):
        from src.clinical_scores import compute_mews
        _, X, _ = small_data
        scores = compute_mews(X[:,4], X[:,3], X[:,5], X[:,2])
        assert ((scores >= 0) | np.isnan(scores)).all()

    def test_esi_range(self, small_data):
        from src.clinical_scores import compute_esi_level
        _, X, _ = small_data
        ages = X[:,0]
        esi  = compute_esi_level(ages, X[:,3], X[:,4], X[:,5], X[:,7], X[:,2])
        valid = esi[~np.isnan(esi)]
        assert (valid >= 1).all() and (valid <= 5).all()


# ─────────────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────────────

class TestMetrics:
    def test_bootstrap_auroc(self, rng):
        from evaluation.metrics import bootstrap_auroc
        y    = (rng.random(500) < 0.25).astype(int)
        prob = np.clip(y * 0.5 + rng.normal(0, 0.25, 500), 0, 1)
        res  = bootstrap_auroc(y, prob, n_bootstrap=200, seed=42)
        assert 0.5 < res["auroc"] < 1.0
        assert res["ci_lo"] < res["auroc"] < res["ci_hi"]

    def test_delong_same_model_p_not_small(self, rng):
        """Comparing a model with itself should give p ≈ 1."""
        from evaluation.metrics import delong_test
        y    = (rng.random(300) < 0.25).astype(int)
        prob = np.clip(y * 0.5 + rng.normal(0, 0.25, 300), 0, 1)
        res  = delong_test(y, prob, prob, n_bootstrap=200, seed=42)
        assert res["delta_auroc"] == pytest.approx(0.0, abs=1e-9)
        assert res["p_value"] > 0.10

    def test_hosmer_lemeshow_perfect(self, rng):
        """Perfect calibration should give high p-value."""
        from evaluation.metrics import hosmer_lemeshow
        # Perfectly calibrated: prob == empirical rate in each bin
        prob = rng.uniform(0, 1, 2000)
        y    = (rng.random(2000) < prob).astype(int)
        res  = hosmer_lemeshow(y, prob, n_bins=10)
        assert res["p_value"] > 0.05   # Should not be rejected

    def test_nri_positive(self, rng):
        """A better model should have positive NRI vs worse model."""
        from evaluation.metrics import nri
        y      = (rng.random(500) < 0.25).astype(int)
        p_good = np.clip(y * 0.6 + rng.normal(0, 0.2, 500), 0.01, 0.99)
        p_bad  = np.clip(y * 0.2 + rng.normal(0, 0.4, 500), 0.01, 0.99)
        res = nri(y, p_good, p_bad, n_bootstrap=200, seed=42)
        assert res["nri"] > 0

    def test_alert_burden_shape(self, rng):
        from evaluation.metrics import alert_burden
        y    = (rng.random(200) < 0.22).astype(int)
        prob = rng.uniform(0, 1, 200)
        df   = alert_burden(y, prob, thresholds=[0.15, 0.28, 0.40, 0.50])
        assert len(df) == 4
        assert "alert_rate" in df.columns
        assert "ppv" in df.columns

    def test_youden_metrics_at_threshold(self, rng):
        from evaluation.metrics import youden_metrics
        y    = (rng.random(400) < 0.22).astype(int)
        prob = np.clip(y * 0.5 + rng.normal(0, 0.25, 400), 0, 1)
        res  = youden_metrics(y, prob, threshold=0.28, n_bootstrap=100, seed=42)
        assert 0 <= res["sensitivity"] <= 1
        assert 0 <= res["specificity"] <= 1
        assert 0 <= res["ppv"] <= 1
        assert res["threshold"] == pytest.approx(0.28)

    def test_interaction_contrast_null(self, rng):
        """When groups are the same model, interaction should be near zero."""
        from evaluation.metrics import interaction_contrast
        y    = (rng.random(400) < 0.25).astype(int)
        p    = np.clip(y * 0.5 + rng.normal(0, 0.25, 400), 0.01, 0.99)
        mask = rng.random(400) > 0.5
        res  = interaction_contrast(y, p, p, mask, n_bootstrap=200, seed=42)
        assert abs(res["delta_diff"]) < 0.05


# ─────────────────────────────────────────────────────────────────────
# Fusion model integration smoke test
# ─────────────────────────────────────────────────────────────────────

class TestFusionModelSmoke:
    def test_predict_proba_shape(self, small_data):
        """DeepTriageCN.predict_proba returns (N,) in [0,1]."""
        from src.fusion_model import DeepTriageCN
        y, X, texts = small_data
        model = DeepTriageCN()
        model.fit(texts, X, y)
        probs = model.predict_proba(texts, X)
        assert probs.shape == (len(y),)
        assert (probs >= 0).all() and (probs <= 1).all()

    def test_fit_predict_better_than_random(self, small_data, rng):
        from src.fusion_model import DeepTriageCN
        from sklearn.metrics import roc_auc_score
        y, X, texts = small_data
        model = DeepTriageCN()
        model.fit(texts, X, y)
        probs = model.predict_proba(texts, X)
        auroc = roc_auc_score(y, probs)
        assert auroc > 0.55, f"Expected AUROC > 0.55, got {auroc:.4f}"

    def test_save_load_roundtrip(self, small_data, tmp_path):
        from src.fusion_model import DeepTriageCN
        import numpy as np
        y, X, texts = small_data
        model = DeepTriageCN()
        model.fit(texts, X, y)
        p1 = model.predict_proba(texts, X)
        path = str(tmp_path / "model.pkl")
        model.save_to_file(path)
        model2 = DeepTriageCN.load_from_file(path)
        p2 = model2.predict_proba(texts, X)
        np.testing.assert_allclose(p1, p2, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
