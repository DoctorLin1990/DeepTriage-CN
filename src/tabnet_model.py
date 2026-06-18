#!/usr/bin/env python3
"""
tabnet_model.py

Thin wrapper around pytorch-tabnet's TabNetClassifier, implementing the
hyperparameters reported in Section 3.6 of the paper.

Final hyperparameters (optimised via grid search):
    N_d = 32, N_a = 32, N_steps = 5, γ = 1.5, λ_sparse = 0.0001
    Optimizer : AdamW
    Batch size : 256, virtual batch size : 128
    LR schedule : cosine annealing

Data contract (matches train_all_models.py):
    TabNet receives *raw* (unstandardised) structured features.
    Its internal Ghost Batch Normalisation handles scale differences.
    Do NOT pre-standardise inputs before calling fit() / predict_proba().
"""

import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)


class TabNetWrapper:
    """
    Wrapper for pytorch-tabnet TabNetClassifier with paper-aligned defaults.

    Args:
        n_d           : Width of decision step output (paper: 32).
        n_a           : Width of attention embedding (paper: 32).
        n_steps       : Number of sequential attention steps (paper: 5).
        gamma         : Relaxation factor for feature re-usage (paper: 1.5).
        lambda_sparse : Sparsity regularisation coefficient (paper: 0.0001).
        seed          : Random seed for reproducibility.
    """

    def __init__(
        self,
        n_d: int = 32,
        n_a: int = 32,
        n_steps: int = 5,
        gamma: float = 1.5,
        lambda_sparse: float = 0.0001,
        seed: int = 42,
    ):
        self.n_d           = n_d
        self.n_a           = n_a
        self.n_steps       = n_steps
        self.gamma         = gamma
        self.lambda_sparse = lambda_sparse
        self.seed          = seed
        self._model        = None

    def _build_model(self) -> None:
        try:
            from pytorch_tabnet.tab_model import TabNetClassifier
        except ImportError as exc:
            raise ImportError(
                "pytorch-tabnet is required.  "
                "Install it with: pip install pytorch-tabnet"
            ) from exc

        self._model = TabNetClassifier(
            n_d=self.n_d,
            n_a=self.n_a,
            n_steps=self.n_steps,
            gamma=self.gamma,
            lambda_sparse=self.lambda_sparse,
            optimizer_fn=__import__("torch").optim.AdamW,
            scheduler_fn=__import__("torch").optim.lr_scheduler.CosineAnnealingLR,
            scheduler_params={"T_max": 50, "eta_min": 1e-5},
            seed=self.seed,
            verbose=0,
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        eval_set: Optional[List[Tuple[np.ndarray, np.ndarray]]] = None,
        max_epochs: int = 200,
        batch_size: int = 256,
        virtual_batch_size: int = 128,
        patience: int = 20,
    ) -> "TabNetWrapper":
        """
        Train TabNet on raw (unstandardised) structured features.

        Args:
            X_train            : Raw features, shape (n_train, 8). May contain NaN;
                                 TabNet handles missing values internally.
            y_train            : Binary labels.
            eval_set           : List of (X_val, y_val) tuples for early stopping.
            max_epochs         : Maximum training epochs.
            batch_size         : Mini-batch size (paper: 256).
            virtual_batch_size : Ghost BN virtual batch size (paper: 128).
            patience           : Early-stopping patience.

        Returns:
            self
        """
        if self._model is None:
            self._build_model()

        # TabNet requires float32 and does not accept NaN out of the box;
        # replace NaN with 0 (neutral after BN) as a safe fallback.
        X_train = np.nan_to_num(np.asarray(X_train, dtype=np.float32), nan=0.0)
        y_train = np.asarray(y_train, dtype=np.int64)

        eval_set_clean = None
        if eval_set:
            eval_set_clean = [
                (np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0),
                 np.asarray(y, dtype=np.int64))
                for X, y in eval_set
            ]

        logger.info(
            f"Training TabNet (n_d={self.n_d}, n_steps={self.n_steps}) "
            f"on {X_train.shape[0]} samples …"
        )
        self._model.fit(
            X_train, y_train,
            eval_set=eval_set_clean,
            eval_name=["val"],
            eval_metric=["auc"],
            max_epochs=max_epochs,
            patience=patience,
            batch_size=batch_size,
            virtual_batch_size=virtual_batch_size,
            drop_last=False,
        )
        logger.info("TabNet training complete.")
        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities for raw structured features.

        Args:
            X : Raw (unstandardised) features, shape (n, 8). NaN-safe.

        Returns:
            np.ndarray of shape (n, 2) — [:, 1] gives P(admission).
        """
        X = np.nan_to_num(np.asarray(X, dtype=np.float32), nan=0.0)
        return self._model.predict_proba(X)

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:
        """Save the TabNet model to *path* (a directory)."""
        Path(path).mkdir(parents=True, exist_ok=True)
        self._model.save_model(str(Path(path) / "tabnet_model"))
        logger.info(f"TabNet saved → {path}")

    def load(self, path: str) -> None:
        """Load a previously saved TabNet model from *path*."""
        if self._model is None:
            self._build_model()
        self._model.load_model(str(Path(path) / "tabnet_model.zip"))
        logger.info(f"TabNet loaded ← {path}")
