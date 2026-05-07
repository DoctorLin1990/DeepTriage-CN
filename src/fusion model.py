#!/usr/bin/env python3
"""
fusion_model.py

Implementation of the DeepTriage-CN late-fusion multimodal framework
(Section 3.5, Figure 1).

Architecture:
    - Text Tower: Frozen BERT-Chinese extracts a 768-d [CLS] embedding.
    - Structured Tower: 8 standardized numerical features (age, sex, 6 vitals).
    - Fusion: Concatenation into a 776-dimensional vector.
    - Classifier: XGBoost with hyperparameters from Section 3.5 and config.yaml.

The model outputs a continuous probability of hospital admission.
"""

import numpy as np
import xgboost as xgb
import joblib
from typing import Optional, Tuple
import logging
from pathlib import Path

from .text_encoder import TextEncoder
from .structured_encoder import StructuredEncoder

logger = logging.getLogger(__name__)


class DeepTriageCN:
    """
    DeepTriage-CN: Multimodal late-fusion model for ED admission prediction.

    Combines frozen BERT embeddings of chief complaint with standardized
    vital signs, using an XGBoost classifier.

    Attributes:
        text_encoder: Frozen BERT-Chinese model.
        structured_encoder: StandardScaler for tabular features.
        classifier: XGBoost classifier instance (after fitting).
        config: Dictionary of XGBoost hyperparameters.
    """

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        structured_encoder: Optional[StructuredEncoder] = None,
        xgb_params: Optional[dict] = None,
        random_state: int = 42
    ):
        """
        Initialize the DeepTriage-CN model.

        Args:
            text_encoder: Instance of TextEncoder. If None, creates default.
            structured_encoder: Instance of StructuredEncoder. If None, creates default.
            xgb_params: Dictionary of XGBoost parameters. Uses paper defaults if None.
            random_state: Random seed for reproducibility.
        """
        self.text_encoder = text_encoder or TextEncoder()
        self.structured_encoder = structured_encoder or StructuredEncoder()

        # Default XGBoost parameters from the paper (Section 3.5) and config.yaml
        self.default_xgb_params = {
            "n_estimators": 200,
            "learning_rate": 0.05,
            "max_depth": 6,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "random_state": random_state,
            "tree_method": "hist",       # Efficient for moderate datasets
            "verbosity": 0,
            # scale_pos_weight will be set during training based on class imbalance
        }

        self.xgb_params = xgb_params or self.default_xgb_params.copy()
        self.classifier = None
        self.is_fitted = False

    def _fuse_features(
        self,
        text_embeddings: np.ndarray,
        structured_features: np.ndarray
    ) -> np.ndarray:
        """
        Concatenate text embeddings and standardized structured features.

        Args:
            text_embeddings: (n_samples, 768) BERT [CLS] embeddings.
            structured_features: (n_samples, 8) standardized numerical features.

        Returns:
            np.ndarray of shape (n_samples, 776).
        """
        return np.hstack([text_embeddings, structured_features])

    def fit(
        self,
        train_texts: list,
        X_train_structured: np.ndarray,
        y_train: np.ndarray,
        scale_pos_weight: Optional[float] = None
    ) -> "DeepTriageCN":
        """
        Train the DeepTriage-CN model.

        Steps:
        1. Extract frozen BERT embeddings for training texts.
        2. Fit and transform structured features using StandardScaler.
        3. Fuse embeddings and structured features.
        4. Train XGBoost classifier.

        Args:
            train_texts: List of chief complaint strings (training set).
            X_train_structured: Raw structured features (n_train, 8), before
                             standardization. May contain NaNs.
            y_train: Binary admission labels (0 or 1).
            scale_pos_weight: Weight for positive class to handle imbalance.
                               If None, calculated as (neg/pos).

        Returns:
            self for method chaining.
        """
        logger.info("Training DeepTriage-CN...")

        # ---- Step 1: Text embeddings (frozen BERT) ----
        logger.info("  [1/4] Extracting text embeddings...")
        text_embeddings = self.text_encoder.encode(train_texts)
        logger.info(f"  Text embeddings shape: {text_embeddings.shape}")

        # ---- Step 2: Structured feature standardization ----
        logger.info("  [2/4] Standardizing structured features...")
        X_structured_std = self.structured_encoder.fit_transform(X_train_structured)
        logger.info(f"  Structured features shape: {X_structured_std.shape}")

        # ---- Step 3: Feature fusion ----
        logger.info("  [3/4] Fusing features...")
        X_fused = self._fuse_features(text_embeddings, X_structured_std)
        logger.info(f"  Fused feature shape: {X_fused.shape}")

        # ---- Step 4: XGBoost training ----
        logger.info("  [4/4] Training XGBoost classifier...")
        if scale_pos_weight is None:
            # Automatically handle class imbalance (Section 3.5)
            n_neg = np.sum(y_train == 0)
            n_pos = np.sum(y_train == 1)
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            logger.info(f"  Calculated scale_pos_weight: {scale_pos_weight:.2f}")

        self.xgb_params["scale_pos_weight"] = scale_pos_weight

        self.classifier = xgb.XGBClassifier(**self.xgb_params)
        self.classifier.fit(X_fused, y_train)
        self.is_fitted = True

        logger.info("DeepTriage-CN training completed.")
        return self

    def predict_proba(self, texts: list, X_structured: np.ndarray) -> np.ndarray:
        """
        Predict admission probabilities for new encounters.

        Args:
            texts: List of chief complaint strings.
            X_structured: Raw structured features (n_samples, 8), pre-standardization.

        Returns:
            np.ndarray of shape (n_samples,) with probabilities of admission.
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction.")

        # Text embeddings
        text_emb = self.text_encoder.encode(texts)

        # Standardize structured features using pre-fitted scaler
        X_structured_std = self.structured_encoder.transform(X_structured)

        # Fuse
        X_fused = self._fuse_features(text_emb, X_structured_std)

        # Predict (returns shape (n_samples, 2); take column 1 for positive class)
        proba = self.classifier.predict_proba(X_fused)[:, 1]
        return proba

    def predict(self, texts: list, X_structured: np.ndarray,
                threshold: float = 0.28) -> np.ndarray:
        """
        Binary prediction at a given threshold (default Youden-optimal = 0.28).

        Args:
            texts: List of chief complaint strings.
            X_structured: Raw structured features.
            threshold: Probability threshold for positive prediction.

        Returns:
            np.ndarray of binary predictions.
        """
        proba = self.predict_proba(texts, X_structured)
        return (proba >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        state = {
            "xgb_params": self.xgb_params,
            "classifier": self.classifier,
            "structured_encoder": self.structured_encoder,
        }
        joblib.dump(state, path)
        logger.info(f"Model saved to {path}")

    def load(self, path: str) -> None:
        """Load a trained model from disk."""
        state = joblib.load(path)
        self.xgb_params = state["xgb_params"]
        self.classifier = state["classifier"]
        self.structured_encoder = state["structured_encoder"]
        self.is_fitted = True
        logger.info(f"Model loaded from {path}")