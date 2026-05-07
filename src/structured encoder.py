#!/usr/bin/env python3
"""
structured_encoder.py

Module for handling structured (tabular) features at triage.
Provides a wrapper around StandardScaler that fits on training data and
transforms validation data, ensuring no label leakage (Section 3.4).

Also handles the concatenation of standardized vital signs with other
non-standardized features (e.g., sex, which is binary and not standardized
in our pipeline, though the paper standardizes all 8 features).

Note: The preprocess.py module already contains the core standardization
logic. This module is provided as a clean, standalone interface consistent
with the repository's modular design.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# The 8 features used in the structured tower (Section 3.5)
STRUCTURED_FEATURE_NAMES = [
    "age", "sex", "temperature", "heart_rate",
    "respiratory_rate", "sbp", "dbp", "spo2"
]


class StructuredEncoder:
    """
    Encoder for tabular features (demographics + vital signs).

    Applies z-score normalization using training-set statistics.
    Missing values are passed through as NaN (XGBoost handles them natively).
    """

    def __init__(self):
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X_train: np.ndarray) -> "StructuredEncoder":
        """
        Fit the StandardScaler on training data.

        Args:
            X_train: Training array of shape (n_samples, 8).

        Returns:
            self for method chaining.
        """
        # StandardScaler can handle NaN by ignoring them during fit
        self.scaler.fit(X_train)
        self._fitted = True
        logger.info(f"StructuredEncoder fitted on {X_train.shape[0]} samples.")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Transform data using fitted scaler.

        Args:
            X: Input array of shape (n_samples, 8).

        Returns:
            Standardized array of same shape.
        """
        if not self._fitted:
            raise RuntimeError("StructuredEncoder must be fitted before transform.")
        return self.scaler.transform(X)

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed training data."""
        return self.fit(X_train).transform(X_train)

    def get_feature_names(self) -> list:
        """Return the list of feature names in order."""
        return STRUCTURED_FEATURE_NAMES.copy()