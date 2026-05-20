#!/usr/bin/env python3
"""
baseline_models.py

Baseline comparator models described in Section 3.6 of the paper:

    VitalsOnlyXGBoost           – same XGBoost classifier without text (Section 3.6)
    RandomForestBaseline        – Random Forest ensemble (Section 3.6)
    TextOnlyLogisticRegression  – Logistic Regression on 768-d BERT embeddings (Section 3.6, ablation)

All three models receive *pre-standardised* structured features
(from preprocess.preprocess_structured), unlike DeepTriage-CN and TabNet
which standardise internally.

Data contract (matches train_all_models.py):
    fit(X_train_std, y_train)       — standardised arrays
    predict_proba(X_val_std)        — standardised arrays
"""

import numpy as np
import logging
from typing import Optional

import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)


class VitalsOnlyXGBoost:
    """
    XGBoost trained on structured (vital signs + demographics) features only.

    Hyperparameters mirror the structured tower of DeepTriage-CN
    (Section 3.5) to ensure a fair ablation comparison.

    Args:
        n_estimators      : Number of boosting rounds (paper: 200).
        learning_rate     : Step size shrinkage (paper: 0.05).
        max_depth         : Maximum tree depth (paper: 6).
        subsample         : Row sub-sampling ratio (paper: 0.8).
        colsample_bytree  : Column sub-sampling ratio (paper: 0.8).
        random_state      : RNG seed.
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
        self._params = dict(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            tree_method="hist",
            verbosity=0,
        )
        self._model: Optional[xgb.XGBClassifier] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "VitalsOnlyXGBoost":
        n_neg = int(np.sum(y_train == 0))
        n_pos = int(np.sum(y_train == 1))
        spw   = n_neg / n_pos if n_pos > 0 else 1.0
        params = {**self._params, "scale_pos_weight": spw}
        self._model = xgb.XGBClassifier(**params)
        self._model.fit(X_train, y_train)
        logger.info(f"VitalsOnlyXGBoost trained (scale_pos_weight={spw:.3f}).")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class RandomForestBaseline:
    """
    Random Forest baseline (Section 3.6).

    Args:
        n_estimators : Number of trees (default 200).
        max_depth    : Maximum tree depth (default 6; mirrors XGBoost config).
        random_state : RNG seed.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 6,
        random_state: int = 42,
    ):
        self._model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "RandomForestBaseline":
        self._model.fit(X_train, y_train)
        logger.info("RandomForestBaseline trained.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)


class TextOnlyLogisticRegression:
    """
    Logistic Regression trained on 768-d frozen BERT embeddings only (Section 3.6).

    Used for ablation: tests whether chief complaint text alone can predict
    admission without any structured vital signs.  The low AUROC (0.712)
    reported in the paper confirms that text alone is insufficient.

    Args:
        C            : Inverse regularisation strength (default 1.0).
        max_iter     : Maximum solver iterations.
        random_state : RNG seed.
    """

    def __init__(
        self,
        C: float = 1.0,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        self._model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight="balanced",
            random_state=random_state,
            solver="lbfgs",
        )

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> "TextOnlyLogisticRegression":
        """
        Args:
            X_train : BERT [CLS] embeddings, shape (n_train, 768).
            y_train : Binary labels.
        """
        self._model.fit(X_train, y_train)
        logger.info("TextOnlyLogisticRegression trained.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Args:
            X : BERT [CLS] embeddings, shape (n, 768).
        Returns:
            np.ndarray shape (n, 2) — [:, 1] gives P(admission).
        """
        return self._model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
