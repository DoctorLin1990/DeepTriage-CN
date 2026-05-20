#!/usr/bin/env python3
"""
preprocess.py

Data loading, cleaning, and temporal splitting for the DeepTriage-CN project.
Implements the cohort selection logic described in Section 3.2 and the
preprocessing steps from Section 3.4 of the paper.

Key features:
    - Temporal split (8,000 training / 2,000 validation)
    - Z-score normalisation with training-set-derived parameters
    - Handling of missing chief complaint text
    - Stratification metadata for subgroup analysis (age ≥ 65)

DATA CONTRACT
-------------
preprocess_structured() standardises the data and returns (X_train_std,
X_val_std, scaler).  The standardised arrays are intended for baseline models
(TabNet, VitalsOnlyXGBoost, RandomForest) that do **not** perform internal
normalisation.

DeepTriageCN performs its own internal standardisation via StructuredEncoder.
Therefore the *raw* (unstandardised) arrays from train_df / val_df must be
passed to DeepTriageCN.fit() and DeepTriageCN.predict_proba():

    X_train_raw = train_df[STRUCTURED_FEATURES].values   # for DeepTriageCN
    X_val_raw   = val_df[STRUCTURED_FEATURES].values     # for DeepTriageCN
    X_train_std, X_val_std, scaler = preprocess_structured(train_df, val_df)
                                                          # for other models
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

# Feature names matching data/feature_names.json
STRUCTURED_FEATURES = [
    "age", "sex", "temperature", "heart_rate",
    "respiratory_rate", "sbp", "dbp", "spo2",
]
TEXT_FEATURE = "chief_complaint"
TARGET = "hospital_admission"


def load_and_split_data(
    data_path: str,
    training_cutoff_date: str = "2024-01-01",
    validation_start_date: str = "2024-01-01",
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and perform a temporal train / validation split.

    Args:
        data_path             : Path to the CSV file containing ED visits.
        training_cutoff_date  : Visits *before* this date → training set.
        validation_start_date : Visits *on or after* this date → validation.
        random_state          : RNG seed for within-split shuffling.

    Returns:
        train_df : Training DataFrame  (~8 000 encounters in the paper).
        val_df   : Validation DataFrame (~2 000 encounters in the paper).
    """
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df["visit_date"] = pd.to_datetime(df["visit_date"])
    logger.info(f"Loaded {len(df)} total encounters from '{data_path}'.")

    # --- First-visit-only deduplication (Section 3.2) ---
    # The paper retains only the first ED visit per patient to ensure
    # independence.  With real EHR data, uncomment the line below and
    # ensure a 'patient_id' column is present.
    # df = df.sort_values("visit_date").drop_duplicates(
    #     subset="patient_id", keep="first"
    # )

    # --- Temporal split ---
    train_df = df[df["visit_date"] < training_cutoff_date].copy()
    val_df   = df[df["visit_date"] >= validation_start_date].copy()

    # Shuffle within each split to remove ordering bias
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(
        drop=True
    )

    logger.info(f"Training   set : {len(train_df)} encounters "
                f"(admission rate {train_df[TARGET].mean():.3f})")
    logger.info(f"Validation set : {len(val_df)} encounters "
                f"(admission rate {val_df[TARGET].mean():.3f})")
    return train_df, val_df


def get_raw_structured(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract *raw* (unstandardised) structured feature arrays.

    These arrays are intended exclusively for DeepTriageCN, which applies
    its own internal NaN-safe z-score normalisation via StructuredEncoder.

    Args:
        train_df : Training DataFrame.
        val_df   : Validation DataFrame.

    Returns:
        X_train_raw : shape (n_train, 8), may contain NaN.
        X_val_raw   : shape (n_val,   8), may contain NaN.
    """
    X_train_raw = train_df[STRUCTURED_FEATURES].values.astype(float)
    X_val_raw   = val_df[STRUCTURED_FEATURES].values.astype(float)
    return X_train_raw, X_val_raw


def preprocess_structured(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, "StructuredEncoder"]:
    """
    Standardise structured features for baseline models (TabNet, XGBoost, RF).

    Standardisation parameters (mean, std) are derived *exclusively* from the
    training set to prevent label leakage (Section 3.4).  NaN values are
    preserved and handled natively by each downstream model.

    Do NOT pass the returned arrays to DeepTriageCN — use get_raw_structured()
    instead (see module-level DATA CONTRACT note).

    Args:
        train_df : Training DataFrame.
        val_df   : Validation DataFrame.

    Returns:
        X_train_std : Standardised training features, shape (n_train, 8).
        X_val_std   : Standardised validation features, shape (n_val,   8).
        encoder     : Fitted StructuredEncoder (carry forward for inference).
    """
    from .structured_encoder import StructuredEncoder

    X_train_raw = train_df[STRUCTURED_FEATURES].values.astype(float)
    X_val_raw   = val_df[STRUCTURED_FEATURES].values.astype(float)

    encoder = StructuredEncoder()
    X_train_std = encoder.fit_transform(X_train_raw)
    X_val_std   = encoder.transform(X_val_raw)

    logger.info(
        f"Structured features standardised — "
        f"train {X_train_std.shape}, val {X_val_std.shape}."
    )
    return X_train_std, X_val_std, encoder


def preprocess_text(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare chief complaint text for BERT encoding.

    Replaces NaN / empty strings (≈2.4 % of encounters, Section 3.4) with
    the special token "[MISSING]".

    Args:
        train_df : Training DataFrame.
        val_df   : Validation DataFrame.

    Returns:
        train_texts : pd.Series of preprocessed complaint strings.
        val_texts   : pd.Series of preprocessed complaint strings.
    """
    def _clean(series: pd.Series) -> pd.Series:
        s = series.fillna("").str.strip()
        return s.replace("", "[MISSING]")

    train_texts = _clean(train_df[TEXT_FEATURE])
    val_texts   = _clean(val_df[TEXT_FEATURE])

    logger.info(
        f"Text preprocessing complete — "
        f"[MISSING] rate: train {(train_texts == '[MISSING]').mean():.3f}, "
        f"val {(val_texts == '[MISSING]').mean():.3f}."
    )
    return train_texts, val_texts


def get_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract binary admission labels.

    Returns:
        y_train, y_val : int arrays (0 = discharged, 1 = admitted).
    """
    y_train = train_df[TARGET].values.astype(int)
    y_val   = val_df[TARGET].values.astype(int)
    return y_train, y_val


def get_subgroup_mask(
    df: pd.DataFrame,
    age_threshold: int = 65,
) -> np.ndarray:
    """
    Boolean mask for the geriatric subgroup analysis (Section 4.4).

    Args:
        df            : DataFrame with an 'age' column.
        age_threshold : Cutoff age (inclusive).

    Returns:
        Boolean array, True where age ≥ threshold.
    """
    return (df["age"] >= age_threshold).values
