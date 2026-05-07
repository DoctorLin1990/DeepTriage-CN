#!/usr/bin/env python3
"""
test_preprocess.py

Unit tests for data loading, temporal splitting, standardization,
and text preprocessing logic.
"""

import pytest
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    preprocess_structured,
    preprocess_text,
    get_labels,
    get_subgroup_mask,
)

# ---- Fixtures ----
@pytest.fixture
def sample_data_path(tmp_path):
    """Create a small synthetic CSV for testing."""
    n = 200
    dates_2023 = pd.date_range("2023-06-01", periods=100, freq="D")
    dates_2024 = pd.date_range("2024-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "visit_id": [f"ED{i}" for i in range(n)],
        "age": np.random.triangular(18, 50, 90, n).astype(int),
        "sex": np.random.binomial(1, 0.5, n),
        "temperature": np.random.normal(36.8, 1.0, n),
        "heart_rate": np.random.normal(80, 15, n).astype(int),
        "respiratory_rate": np.random.normal(16, 4, n).astype(int),
        "sbp": np.random.normal(125, 20, n).astype(int),
        "dbp": np.random.normal(78, 12, n).astype(int),
        "spo2": np.random.normal(98, 2, n),
        "chief_complaint": np.random.choice(["胸痛", "呼吸困难", "头晕", "发热"], n),
        "hospital_admission": np.random.binomial(1, 0.22, n),
        "visit_date": list(dates_2023) + list(dates_2024),
    })
    path = tmp_path / "test_data.csv"
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return str(path)


# ---- Tests ----
def test_load_and_split(sample_data_path):
    train, val = load_and_split_data(sample_data_path,
                                     training_cutoff_date="2024-01-01",
                                     validation_start_date="2024-01-01")
    assert len(train) == 100, f"Expected 100 training, got {len(train)}"
    assert len(val) == 100, f"Expected 100 validation, got {len(val)}"
    assert all(train["visit_date"] < pd.Timestamp("2024-01-01"))
    assert all(val["visit_date"] >= pd.Timestamp("2024-01-01"))


def test_preprocess_structured(sample_data_path):
    train, val = load_and_split_data(sample_data_path)
    X_train, X_val, scaler = preprocess_structured(train, val)
    assert X_train.shape == (100, 8), f"Shape mismatch: {X_train.shape}"
    assert X_val.shape == (100, 8)
    # Scaler should be fitted and produce mean ~0, std ~1 for non-NaN
    train_mean = np.nanmean(X_train, axis=0)
    assert all(abs(train_mean) < 0.2), f"Mean too far from zero: {train_mean}"


def test_preprocess_text(sample_data_path):
    train, val = load_and_split_data(sample_data_path)
    # Inject a missing complaint
    train.loc[0, "chief_complaint"] = ""
    train_texts, val_texts = preprocess_text(train, val)
    assert train_texts.iloc[0] == "[MISSING]"
    assert "[MISSING]" not in val_texts.values  # None missing in val


def test_get_labels(sample_data_path):
    train, val = load_and_split_data(sample_data_path)
    y_train, y_val = get_labels(train, val)
    assert y_train.shape == (100,)
    assert y_val.shape == (100,)
    assert set(y_train) <= {0, 1}


def test_subgroup_mask(sample_data_path):
    train, val = load_and_split_data(sample_data_path)
    mask = get_subgroup_mask(val, age_threshold=65)
    assert mask.sum() == (val["age"] >= 65).sum()