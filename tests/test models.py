#!/usr/bin/env python3
"""
test_models.py

Unit tests for model instantiations, fitting, and prediction.
Uses minimal synthetic data to verify that the pipeline runs end-to-end.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.text_encoder import TextEncoder
from src.structured_encoder import StructuredEncoder
from src.fusion_model import DeepTriageCN
from src.baseline_models import VitalsOnlyXGBoost, RandomForestBaseline
from src.robustness import simulate_degraded_data
from src.structured_encoder import STRUCTURED_FEATURE_NAMES


@pytest.fixture
def dummy_data():
    n = 50
    texts = ["胸痛", "呼吸困难", "发热", "头晕"] * (n // 4 + 1)
    texts = texts[:n]
    X_str = np.random.randn(n, 8).astype(np.float32)
    y = np.random.binomial(1, 0.3, n)
    return texts, X_str, y


def test_text_encoder():
    encoder = TextEncoder(model_name="bert-base-chinese", max_length=32)
    texts = ["胸痛", "呼吸困难", "[MISSING]"]
    emb = encoder.encode(texts)
    assert emb.shape == (3, 768)
    # Check that missing token produces a finite embedding
    assert np.isfinite(emb[2]).all()


def test_structured_encoder():
    enc = StructuredEncoder()
    X = np.random.randn(20, 8)
    X_trans = enc.fit_transform(X)
    assert X_trans.shape == (20, 8)
    # Same data twice should be equal after transformation
    X_trans2 = enc.transform(X)
    np.testing.assert_array_almost_equal(X_trans, X_trans2)


def test_deeptriage_cn_fit_predict(dummy_data):
    texts, X_str, y = dummy_data
    model = DeepTriageCN()
    model.fit(texts, X_str, y)
    proba = model.predict_proba(texts[:5], X_str[:5])
    assert proba.shape == (5,)
    assert np.all((proba >= 0) & (proba <= 1))


def test_vitals_only_xgb(dummy_data):
    _, X_str, y = dummy_data
    model = VitalsOnlyXGBoost()
    model.fit(X_str, y)
    proba = model.predict_proba(X_str[:5])
    assert proba.shape == (5,)


def test_random_forest(dummy_data):
    _, X_str, y = dummy_data
    model = RandomForestBaseline()
    model.fit(X_str, y)
    proba = model.predict_proba(X_str[:5])
    assert proba.shape == (5,)


def test_simulate_degraded_data():
    X = np.random.randn(100, 8)
    degraded_sets = simulate_degraded_data(
        X, STRUCTURED_FEATURE_NAMES,
        missing_proportions=[0.2],
        noise_sigma=0.5,
    )
    X_deg = degraded_sets[0.2]
    assert X_deg.shape == (100, 8)
    # Check that some values became NaN
    assert np.any(np.isnan(X_deg))