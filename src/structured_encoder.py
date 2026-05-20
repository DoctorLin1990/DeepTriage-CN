#!/usr/bin/env python3
"""
structured_encoder.py

Module for handling structured (tabular) features at triage.
Provides a NaN-safe z-score normalizer that fits on training data and
transforms validation data, ensuring no label leakage (Section 3.4).

Bug fixed (L6): sklearn.StandardScaler raises ValueError on NaN inputs.
This module uses a custom NaN-aware implementation:
    - Mean and std are computed with np.nanmean / np.nanstd
    - NaN positions remain NaN after standardization
    - XGBoost handles residual NaN natively via sparsity-aware split-finding
"""

import numpy as np
import pandas as pd
from typing import Optional
import logging

logger = logging.getLogger(__name__)

# The 8 structured features used in the model (Section 3.5, Table 1)
STRUCTURED_FEATURE_NAMES = [
    "age", "sex", "temperature", "heart_rate",
    "respiratory_rate", "sbp", "dbp", "spo2"
]


class StructuredEncoder:
    """
    NaN-aware z-score normalizer for tabular vital signs and demographics.

    Parameters are derived exclusively from training data to prevent leakage.
    Missing values (NaN) pass through unchanged and are handled downstream
    by XGBoost's sparsity-aware algorithm.

    Attributes:
        mean_  : np.ndarray of shape (n_features,), training column means.
        std_   : np.ndarray of shape (n_features,), training column stds.
        _fitted: bool, whether the encoder has been fitted.
    """

    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None
        self._fitted: bool = False

    def fit(self, X_train: np.ndarray) -> "StructuredEncoder":
        """
        Compute column-wise mean and std from training data, ignoring NaN.

        Args:
            X_train: Raw training array of shape (n_samples, n_features).
                     May contain NaN values.

        Returns:
            self for method chaining.
        """
        X = np.asarray(X_train, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        self.std_ = np.nanstd(X, axis=0, ddof=0)
        # Prevent division-by-zero for constant or binary features (e.g., sex)
        self.std_[self.std_ == 0] = 1.0
        self._fitted = True
        logger.info(
            f"StructuredEncoder fitted on {X.shape[0]} samples, "
            f"{X.shape[1]} features."
        )
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization, preserving NaN positions.

        Args:
            X: Input array of shape (n_samples, n_features).

        Returns:
            Standardized array of same shape. NaN entries remain NaN.
        """
        if not self._fitted:
            raise RuntimeError("StructuredEncoder must be fitted before transform.")
        X = np.asarray(X, dtype=float).copy()
        X_std = (X - self.mean_) / self.std_
        # NaN propagates naturally through subtraction/division
        return X_std

    def fit_transform(self, X_train: np.ndarray) -> np.ndarray:
        """Fit on training data and return transformed training data."""
        return self.fit(X_train).transform(X_train)

    def get_feature_names(self) -> list:
        """Return the list of structured feature names in order."""
        return STRUCTURED_FEATURE_NAMES.copy()

    def get_params(self) -> dict:
        """Return a serialisable dict of the fitted parameters."""
        return {
            "mean_": self.mean_.tolist() if self.mean_ is not None else None,
            "std_":  self.std_.tolist()  if self.std_  is not None else None,
            "_fitted": self._fitted,
        }

    def set_params(self, params: dict) -> None:
        """Restore fitted parameters from a serialised dict (used by load)."""
        if params["mean_"] is not None:
            self.mean_ = np.array(params["mean_"])
            self.std_  = np.array(params["std_"])
            self._fitted = params["_fitted"]
