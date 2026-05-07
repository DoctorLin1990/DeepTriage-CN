#!/usr/bin/env python3
"""
baseline_models.py

Implementations of the three comparator models evaluated in the paper
(Section 3.6):
    1. Vitals-only XGBoost (structured tower alone, identical hyperparameters)
    2. Random Forest (widely used ensemble baseline)
    3. Text-only Logistic Regression using frozen BERT embeddings

These models allow ablation analysis to isolate the contribution of
each modality.
"""

import numpy as np
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class VitalsOnlyXGBoost:
    """
    XGBoost classifier trained exclusively on the 8 structured features
    (age, sex, 6 vital signs). Uses the same hyperparameters as the
    XGBoost head of DeepTriage-CN (Section 3.6).
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 6,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        self.model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            verbosity=0,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "VitalsOnlyXGBoost":
        """
        Train vitals-only XGBoost.
        Handles missing values natively via XGBoost's sparsity-aware split.
        Standardization is applied for consistency but not strictly required by XGBoost.
        """
        # Compute scale_pos_weight
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()
        self.model.set_params(scale_pos_weight=n_neg / n_pos if n_pos > 0 else 1.0)

        X_std = self.scaler.fit_transform(X)
        self.model.fit(X_std, y)
        self._fitted = True
        logger.info("Vitals-only XGBoost fitted.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X_std = self.scaler.transform(X)
        return self.model.predict_proba(X_std)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.28) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class RandomForestBaseline:
    """Random Forest classifier (Section 3.6) for tabular data."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 6,
                 random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1,
        )
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestBaseline":
        # For RF, we impute missing values with 0 after standardization
        X_std = self.scaler.fit_transform(X)
        X_std = np.nan_to_num(X_std, nan=0.0)
        self.model.fit(X_std, y)
        self._fitted = True
        logger.info("Random Forest fitted.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        X_std = self.scaler.transform(X)
        X_std = np.nan_to_num(X_std, nan=0.0)
        return self.model.predict_proba(X_std)[:, 1]

    def predict(self, X: np.ndarray, threshold: float = 0.28) -> np.ndarray:
        return (self.predict_proba(X) >= threshold).astype(int)


class TextOnlyLogisticRegression:
    """
    Logistic Regression trained exclusively on frozen BERT embeddings
    of chief complaint text. Used for ablation to isolate text contribution.
    """

    def __init__(self, random_state: int = 42):
        self.model = LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )
        self._fitted = False

    def fit(self, text_embeddings: np.ndarray, y: np.ndarray) -> "TextOnlyLogisticRegression":
        self.model.fit(text_embeddings, y)
        self._fitted = True
        logger.info("Text-only Logistic Regression fitted.")
        return self

    def predict_proba(self, text_embeddings: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self.model.predict_proba(text_embeddings)[:, 1]

    def predict(self, text_embeddings: np.ndarray, threshold: float = 0.28) -> np.ndarray:
        return (self.predict_proba(text_embeddings) >= threshold).astype(int)