#!/usr/bin/env python3
"""
tabnet_model.py

TabNet implementation for the structured-only deep learning comparator,
as described in Section 3.6 of the paper.

Hyperparameters were optimized via grid search (Section 3.6) and set to:
    N_d = 32
    N_a = 32
    N_steps = 5
    gamma = 1.5
    lambda_sparse = 0.0001
    Optimizer: AdamW
    Batch size: 256, Virtual batch size: 128
    Cosine annealing learning rate schedule

The model uses only the 8 structured features (no text).
"""

import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from typing import Optional
import logging
import joblib

logger = logging.getLogger(__name__)


class TabNetWrapper(BaseEstimator, ClassifierMixin):
    """
    Wrapper around pytorch-tabnet's TabNetClassifier to provide a
    scikit-learn compatible interface.

    Hyperparameters are fixed to the values used in the paper (Section 3.6).
    """

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 0.0001,
        optimizer_fn=torch.optim.Adam,
        optimizer_params: dict = None,
        scheduler_fn=torch.optim.lr_scheduler.CosineAnnealingLR,
        scheduler_params: dict = None,
        mask_type: str = "sparsemax",
        seed: int = 42,
        verbose: int = 0,
    ):
        """
        Initialize TabNet with paper-specified hyperparameters.

        Args:
            n_d: Width of decision prediction layer.
            n_a: Width of attention embedding for each mask.
            n_steps: Number of steps in the architecture.
            gamma: Relaxation factor for feature selection.
            lambda_sparse: Sparsity regularization strength.
            optimizer_fn: Optimizer class (AdamW from torch).
            optimizer_params: Dictionary of optimizer parameters (e.g., lr, weight_decay).
            scheduler_fn: LR scheduler class (CosineAnnealingLR).
            scheduler_params: Dictionary of scheduler parameters (e.g., T_max, eta_min).
            mask_type: Type of mask for feature selection.
            seed: Random seed.
            verbose: Verbosity level.
        """
        self.n_d = n_d
        self.n_a = n_a
        self.n_steps = n_steps
        self.gamma = gamma
        self.lambda_sparse = lambda_sparse
        self.optimizer_fn = optimizer_fn
        self.optimizer_params = optimizer_params or {"lr": 2e-2, "weight_decay": 1e-5}
        self.scheduler_fn = scheduler_fn
        self.scheduler_params = scheduler_params or {"T_max": 100, "eta_min": 1e-5}
        self.mask_type = mask_type
        self.seed = seed
        self.verbose = verbose

        self.model = None
        self._fitted = False

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        eval_set: Optional[list] = None,
        max_epochs: int = 100,
        patience: int = 20,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
    ) -> "TabNetWrapper":
        """
        Train the TabNet model.

        Args:
            X: Feature matrix (n_samples, 8), may contain NaNs (handled by TabNet).
            y: Binary labels.
            eval_set: List of (X_val, y_val) for early stopping.
            max_epochs: Maximum number of epochs.
            patience: Early stopping patience.
            batch_size: Training batch size (paper: 256).
            virtual_batch_size: Size for Ghost Batch Normalization (paper: 128).

        Returns:
            self
        """
        self.model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=self.optimizer_fn,
            optimizer_params=self.optimizer_params,
            scheduler_fn=self.scheduler_fn,
            scheduler_params=self.scheduler_params,
            mask_type=self.mask_type,
            seed=self.seed,
            verbose=self.verbose,
        )

        logger.info("Training TabNet (structured-only)...")
        self.model.fit(
            X_train=X,
            y_train=y,
            eval_set=eval_set,
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False,
        )
        self._fitted = True
        logger.info("TabNet training completed.")
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return probability estimates (shape: n_samples x 2)."""
        if not self._fitted:
            raise RuntimeError("Must fit before predict.")
        return self.model.predict_proba(X)

    def predict(self, X: np.ndarray, threshold: float = 0.28) -> np.ndarray:
        """Binary prediction at given threshold."""
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)

    def save(self, path: str) -> None:
        """Save model to disk using joblib."""
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        # TabNet can be saved via its own method, but we also save wrapper state
        self.model.save_model(path)
        # save wrapper params
        joblib.dump(self.__dict__, path + ".wrapper")

    def load(self, path: str) -> None:
        """Load model from disk."""
        self.model = TabNetClassifier()
        self.model.load_model(path)
        wrapper_state = joblib.load(path + ".wrapper")
        for k, v in wrapper_state.items():
            if k != "model":
                setattr(self, k, v)
        self._fitted = True
        logger.info("TabNet model loaded.")