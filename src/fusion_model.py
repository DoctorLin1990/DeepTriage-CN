#!/usr/bin/env python3
"""
fusion_model.py

Implementation of the DeepTriage-CN late-fusion multimodal framework
(Section 3.5, Figure 1).

Architecture:
    Text Tower   : Frozen BERT-Chinese (bert-base-chinese) → 768-d [CLS] vector.
    Structured   : 8 raw vital-sign / demographic features → NaN-safe z-score
                   normalisation via StructuredEncoder → 8-d vector.
    Fusion       : Concatenation → 776-dimensional feature vector.
    Classifier   : XGBoost (hyperparameters from Section 3.5 / config.yaml).

Bugs fixed:
    B3 / B4 – save() stored a plain dict; evaluate_all_models.py then called
              joblib.load() which returned the dict, not a DeepTriageCN instance,
              so subsequent .predict_proba() raised AttributeError.
              Fix: save() now serialises the full model state and the class-
              level factory method load_from_file() rebuilds a ready-to-use
              DeepTriageCN object.

    L1       – DeepTriageCN.fit() received data that had already been
              standardised by preprocess_structured(), then called
              structured_encoder.fit_transform() internally, causing a second
              standardisation and severely distorting the feature distribution.
              Fix: DeepTriageCN always expects *raw* (unstandardised) structured
              features. Callers must NOT pre-standardise the data before passing
              it to fit() / predict_proba().
"""

import numpy as np
import xgboost as xgb
import joblib
from typing import Optional, List
import logging
from pathlib import Path

from .text_encoder import TextEncoder
from .structured_encoder import StructuredEncoder

logger = logging.getLogger(__name__)


class DeepTriageCN:
    """
    DeepTriage-CN: Multimodal late-fusion model for ED admission prediction.

    Combines frozen BERT embeddings of the chief complaint with NaN-safe
    z-score-normalised vital signs, classified by XGBoost.

    IMPORTANT – data contract
    -------------------------
    fit() and predict_proba() expect **raw, unstandardised** structured
    features.  Standardisation is handled internally by StructuredEncoder.
    Do NOT call preprocess_structured() on the arrays before passing them
    to this class.
    """

    def __init__(
        self,
        text_encoder: Optional[TextEncoder] = None,
        structured_encoder: Optional[StructuredEncoder] = None,
        xgb_params: Optional[dict] = None,
        random_state: int = 42,
    ):
        self.text_encoder      = text_encoder      or TextEncoder()
        self.structured_encoder = structured_encoder or StructuredEncoder()
        self.random_state      = random_state

        # Default XGBoost hyperparameters (Section 3.5 / config.yaml)
        self._default_xgb_params = {
            "n_estimators":     200,
            "learning_rate":    0.05,
            "max_depth":        6,
            "subsample":        0.8,
            "colsample_bytree": 0.8,
            "random_state":     random_state,
            "tree_method":      "hist",
            "verbosity":        0,
        }
        self.xgb_params = xgb_params or self._default_xgb_params.copy()
        self.classifier: Optional[xgb.XGBClassifier] = None
        self.is_fitted: bool = False

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse(
        self,
        text_embeddings: np.ndarray,
        structured_std: np.ndarray,
    ) -> np.ndarray:
        """Concatenate (n, 768) and (n, 8) → (n, 776)."""
        return np.hstack([text_embeddings, structured_std])

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_texts: List[str],
        X_train_raw: np.ndarray,
        y_train: np.ndarray,
        scale_pos_weight: Optional[float] = None,
    ) -> "DeepTriageCN":
        """
        Train the DeepTriage-CN model end-to-end.

        Args:
            train_texts    : List of raw chief-complaint strings (training set).
            X_train_raw    : Raw (unstandardised) structured features,
                             shape (n_train, 8).  May contain NaN.
            y_train        : Binary admission labels (0 or 1).
            scale_pos_weight: XGBoost positive-class weight for imbalance.
                              Computed automatically from y_train if None.

        Returns:
            self
        """
        logger.info("Training DeepTriage-CN …")

        # Step 1 – BERT embeddings (frozen)
        logger.info("  [1/4] Encoding text (frozen BERT-Chinese) …")
        text_emb = self.text_encoder.encode(train_texts)
        logger.info(f"        Text embeddings: {text_emb.shape}")

        # Step 2 – Fit-and-transform structured features (NaN-safe)
        logger.info("  [2/4] Fitting StructuredEncoder on training data …")
        X_std = self.structured_encoder.fit_transform(
            np.asarray(X_train_raw, dtype=float)
        )
        logger.info(f"        Structured features: {X_std.shape}")

        # Step 3 – Late fusion
        logger.info("  [3/4] Fusing modalities …")
        X_fused = self._fuse(text_emb, X_std)
        logger.info(f"        Fused feature matrix: {X_fused.shape}")

        # Step 4 – XGBoost
        logger.info("  [4/4] Training XGBoost classifier …")
        if scale_pos_weight is None:
            n_neg = int(np.sum(y_train == 0))
            n_pos = int(np.sum(y_train == 1))
            scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
            logger.info(f"        Auto scale_pos_weight = {scale_pos_weight:.3f}")

        params = self.xgb_params.copy()
        params["scale_pos_weight"] = scale_pos_weight
        self.classifier = xgb.XGBClassifier(**params)
        self.classifier.fit(X_fused, y_train)
        self.xgb_params = params          # persist final params (incl. spw)
        self.is_fitted  = True

        logger.info("DeepTriage-CN training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(
        self,
        texts: List[str],
        X_raw: np.ndarray,
    ) -> np.ndarray:
        """
        Predict admission probability for new encounters.

        Args:
            texts : List of chief-complaint strings.
            X_raw : Raw (unstandardised) structured features, shape (n, 8).

        Returns:
            np.ndarray of shape (n,) with P(admission).
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_proba().")

        text_emb = self.text_encoder.encode(texts)
        X_std    = self.structured_encoder.transform(
            np.asarray(X_raw, dtype=float)
        )
        X_fused  = self._fuse(text_emb, X_std)
        return self.classifier.predict_proba(X_fused)[:, 1]

    def predict(
        self,
        texts: List[str],
        X_raw: np.ndarray,
        threshold: float = 0.28,
    ) -> np.ndarray:
        """
        Binary prediction at the Youden-optimal threshold (default 0.28).

        Args:
            texts     : Chief-complaint strings.
            X_raw     : Raw structured features.
            threshold : Probability cutoff.

        Returns:
            np.ndarray of int (0 or 1).
        """
        return (self.predict_proba(texts, X_raw) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Persistence  (Bug B3 / B4 fix)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """
        Serialise the full model state to disk.

        Saved state includes:
            - xgb_params          : dict of XGBoost hyperparameters
            - classifier          : fitted XGBClassifier
            - structured_encoder  : fitted StructuredEncoder (with mean/std)
            - text_encoder_config : dict with model_name and max_length so the
                                    TextEncoder can be reconstructed on load

        The TextEncoder weights themselves (BERT) are not serialised because
        they are frozen and always loaded from HuggingFace; only the config
        (model_name, max_length) is stored.
        """
        state = {
            "xgb_params":           self.xgb_params,
            "classifier":           self.classifier,
            "structured_encoder_params": self.structured_encoder.get_params(),
            "text_encoder_config": {
                "model_name": self.text_encoder.model_name,
                "max_length": self.text_encoder.max_length,
            },
            "is_fitted": self.is_fitted,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(state, path)
        logger.info(f"DeepTriage-CN saved → {path}")

    def load(self, path: str) -> None:
        """
        Restore model state from disk (in-place).

        After calling this method, the instance is ready for predict_proba().
        A fresh TextEncoder is instantiated from the saved config (model_name,
        max_length) and will produce bit-identical embeddings because BERT
        weights are always loaded frozen from the same HuggingFace checkpoint.
        """
        state = joblib.load(path)
        self.xgb_params  = state["xgb_params"]
        self.classifier  = state["classifier"]
        self.is_fitted   = state.get("is_fitted", True)

        # Restore StructuredEncoder fitted parameters
        enc_params = state.get("structured_encoder_params")
        if enc_params:
            self.structured_encoder = StructuredEncoder()
            self.structured_encoder.set_params(enc_params)
        else:
            # Backward-compat: old format stored the object directly
            self.structured_encoder = state.get("structured_encoder",
                                                 StructuredEncoder())

        # Re-instantiate TextEncoder from saved config
        te_cfg = state.get("text_encoder_config", {})
        self.text_encoder = TextEncoder(
            model_name=te_cfg.get("model_name", "bert-base-chinese"),
            max_length=te_cfg.get("max_length", 64),
        )
        logger.info(f"DeepTriage-CN loaded ← {path}")

    @classmethod
    def load_from_file(cls, path: str) -> "DeepTriageCN":
        """
        Class-level factory: load a saved model and return a ready instance.

        Usage (evaluate_all_models.py, generate_all_figures.py):
            from src.fusion_model import DeepTriageCN
            deeptriage = DeepTriageCN.load_from_file("outputs/models/deeptriage_cn.pkl")
            proba = deeptriage.predict_proba(val_texts, X_val_raw)

        This replaces the previous (broken) pattern of:
            deeptriage = joblib.load(...)   # returned a plain dict
        """
        instance = cls.__new__(cls)
        # Set minimal defaults so load() can assign freely
        instance.structured_encoder = StructuredEncoder()
        instance.text_encoder       = None
        instance.classifier         = None
        instance.xgb_params         = {}
        instance.is_fitted          = False
        instance.random_state       = 42
        instance._default_xgb_params = {}
        instance.load(path)
        return instance
