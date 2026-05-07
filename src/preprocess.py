#!/usr/bin/env python3
"""
preprocess.py

Data loading, cleaning, and temporal splitting for the DeepTriage-CN project.
Implements the cohort selection logic described in Section 3.2 and the
preprocessing steps from Section 3.4 of the paper.

Key features:
    - Temporal split (8,000 training / 2,000 validation)
    - Z-score normalization with training-set-derived parameters
    - Handling of missing chief complaint text
    - Stratification metadata for subgroup analysis (age ≥ 65)
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)

# Feature names matching feature_names.json
STRUCTURED_FEATURES = [
    "age", "sex", "temperature", "heart_rate",
    "respiratory_rate", "sbp", "dbp", "spo2"
]
TEXT_FEATURE = "chief_complaint"
TARGET = "hospital_admission"


def load_and_split_data(
    data_path: str,
    training_cutoff_date: str = "2024-01-01",
    validation_start_date: str = "2024-01-01",
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the dataset and perform temporal train/validation split.

    Args:
        data_path: Path to the CSV file containing ED visits.
        training_cutoff_date: Visits before this date are used for training.
        validation_start_date: Visits on or after this date are used for validation.
        random_state: Random seed for shuffling (applied within splits only).

    Returns:
        train_df: Training DataFrame (8000 encounters in the paper).
        val_df: Temporal validation DataFrame (2000 encounters in the paper).
    """
    df = pd.read_csv(data_path, encoding="utf-8-sig")
    df["visit_date"] = pd.to_datetime(df["visit_date"])

    logger.info(f"Loaded {len(df)} total encounters.")

    # --- First-visit-only logic (Section 3.2) ---
    # The paper states: "To ensure independence, only the first visit
    # per patient was retained."
    # In practice, this requires a patient identifier. With synthetic data,
    # we assume each visit_id is unique. This step is documented here for
    # users applying the code to real EHR data.
    # df = df.sort_values("visit_date").drop_duplicates(subset="patient_id", keep="first")

    # --- Temporal split ---
    train_df = df[df["visit_date"] < training_cutoff_date].copy()
    val_df = df[df["visit_date"] >= validation_start_date].copy()

    # Shuffle within each split to avoid ordering bias
    train_df = train_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
    val_df = val_df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    logger.info(f"Training set: {len(train_df)} encounters")
    logger.info(f"Validation set: {len(val_df)} encounters")
    logger.info(f"Training admission rate: {train_df[TARGET].mean():.3f}")
    logger.info(f"Validation admission rate: {val_df[TARGET].mean():.3f}")

    return train_df, val_df


def preprocess_structured(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    scaler: Optional[StandardScaler] = None
) -> Tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Standardize structured features using z-score normalization.

    As described in Section 3.4: parameters (mean, std) are derived
    exclusively from the training set to prevent label leakage.

    Missing values are preserved as NaN; they will be handled natively
    by XGBoost's sparsity-aware split-finding algorithm (Section 3.4).

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.
        scaler: Pre-fitted StandardScaler. If None, a new one is fitted on train_df.

    Returns:
        X_train: Standardized training features (n_train x 8).
        X_val: Standardized validation features (n_val x 8).
        scaler: The fitted StandardScaler object.
    """
    if scaler is None:
        scaler = StandardScaler()
        # fit only on training data
        scaler.fit(train_df[STRUCTURED_FEATURES])

    X_train = scaler.transform(train_df[STRUCTURED_FEATURES])
    X_val = scaler.transform(val_df[STRUCTURED_FEATURES])

    logger.info(f"Structured features standardized. "
                f"Training shape: {X_train.shape}, "
                f"Validation shape: {X_val.shape}")

    return X_train, X_val, scaler


def preprocess_text(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[pd.Series, pd.Series]:
    """
    Prepare chief complaint text for BERT encoding.

    Handles empty strings (2.4% of encounters per Section 3.4) by replacing
    them with the special token "[MISSING]".

    Args:
        train_df: Training DataFrame.
        val_df: Validation DataFrame.

    Returns:
        train_texts: Series of preprocessed chief complaint strings.
        val_texts: Series of preprocessed chief complaint strings.
    """
    train_texts = train_df[TEXT_FEATURE].fillna("").str.strip()
    val_texts = val_df[TEXT_FEATURE].fillna("").str.strip()

    # Replace empty strings with [MISSING] token (Section 3.4)
    train_texts = train_texts.replace("", "[MISSING]")
    val_texts = val_texts.replace("", "[MISSING]")

    logger.info(f"Text preprocessing complete. "
                f"Missing text in train: {(train_texts == '[MISSING]').mean():.3f}, "
                f"Missing text in val: {(val_texts == '[MISSING]').mean():.3f}")

    return train_texts, val_texts


def get_labels(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract target labels for training and validation sets.

    Returns:
        y_train, y_val: Binary labels (0 = discharged, 1 = admitted).
    """
    y_train = train_df[TARGET].values.astype(int)
    y_val = val_df[TARGET].values.astype(int)
    return y_train, y_val


def get_subgroup_mask(
    df: pd.DataFrame,
    age_threshold: int = 65
) -> np.ndarray:
    """
    Create a boolean mask for the geriatric subgroup (Section 4.4).

    Args:
        df: DataFrame containing an 'age' column.
        age_threshold: Age cutoff for the elderly subgroup.

    Returns:
        Boolean array where True indicates age ≥ threshold.
    """
    return (df["age"] >= age_threshold).values