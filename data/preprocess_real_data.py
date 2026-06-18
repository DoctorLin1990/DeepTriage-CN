#!/usr/bin/env python3
"""
preprocess_real_data.py
=======================
Validates, cleans, and exports the institutional ED dataset for use
in the DeepTriage-CN pipeline.

This script should be run **once** before training. It:
  1. Validates column presence and data types
  2. Applies the column rename mapping from config.yaml
  3. Encodes the sex variable (string → 0/1)
  4. Removes duplicate patient visits (first visit retained; Section 3.2)
  5. Filters to adults (age ≥ 18)
  6. Validates the temporal split (training vs validation dates)
  7. Writes the cleaned CSV to data/ed_visits.csv

Usage
-----
    python data/preprocess_real_data.py \\
        --input  /path/to/raw_institutional_data.csv \\
        --output data/ed_visits.csv \\
        --config config.yaml

Expected input format
---------------------
The institutional CSV must contain at minimum these columns
(names are configurable in config.yaml → column_mapping):

  Patient_ID          : str    — unique patient identifier
  visit_date          : str    — ISO 8601 datetime of triage registration
  Age                 : int    — patient age in years
  Gender              : str    — "Male" or "Female"
  Hospital_Admission  : int    — 1 = admitted, 0 = discharged from ED
  ESI                 : int    — Emergency Severity Index level (1–5)
  Chief_Complaint     : str    — nurse-recorded triage chief complaint
  Temperature         : float  — °C; NaN if not recorded
  Pulse               : float  — bpm; NaN if not recorded
  Respiratory_Rate    : float  — /min; NaN if not recorded
  Systolic_BP         : float  — mmHg; NaN if not recorded
  Diastolic_BP        : float  — mmHg; NaN if not recorded
  SpO2                : float  — %; NaN if not recorded

Notes
-----
- Vital signs that were not recorded at triage should be left as NaN (not
  imputed). The pipeline handles missingness natively via XGBoost's
  sparsity-aware split-finding (Section 3.4).
- Chief complaint text must be in Mandarin Chinese for bert-base-chinese
  to produce meaningful embeddings. For visits where the triage nurse did not
  record a chief complaint (2.4% in this cohort), leave the cell empty; the
  pipeline replaces it with a [MISSING] token automatically.
- The temporal split is applied by visit_date: all visits before
  training_cutoff_date form the training set; those on or after
  validation_start_date form the temporal validation set.
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Required raw columns (before renaming)
REQUIRED_COLS = {
    "Patient_ID", "visit_date", "Age", "Gender",
    "Hospital_Admission", "ESI", "Chief_Complaint",
    "Temperature", "Pulse", "Respiratory_Rate",
    "Systolic_BP", "Diastolic_BP", "SpO2",
}

# Physiological plausibility bounds (values outside these are set to NaN)
BOUNDS = {
    "temperature":      (34.0, 43.0),
    "heart_rate":       (20,   220),
    "respiratory_rate": (4,    60),
    "sbp":              (50,   250),
    "dbp":              (20,   150),
    "spo2":             (50,   100),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="Validate and clean institutional ED dataset for DeepTriage-CN."
    )
    p.add_argument("--input",  required=True,
                   help="Path to raw institutional CSV.")
    p.add_argument("--output", default="data/ed_visits.csv",
                   help="Path for cleaned output CSV (default: data/ed_visits.csv).")
    p.add_argument("--config", default="config.yaml",
                   help="Path to config.yaml (default: config.yaml).")
    p.add_argument("--dry-run", action="store_true",
                   help="Validate only; do not write output.")
    return p.parse_args()


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def validate_columns(df: pd.DataFrame, required: set) -> None:
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Input CSV is missing {len(missing)} required column(s): "
            f"{sorted(missing)}\n"
            f"Please check the column_mapping section of config.yaml."
        )
    logger.info("  Column validation: PASSED (%d columns present)", len(df.columns))


def apply_column_mapping(df: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename = {k: v for k, v in mapping.items() if k in df.columns}
    df = df.rename(columns=rename)
    logger.info("  Renamed %d columns.", len(rename))
    return df


def encode_sex(df: pd.DataFrame, positive_value: str = "Male") -> pd.DataFrame:
    if df["sex"].dtype == object:
        df["sex"] = (df["sex"].str.strip() == positive_value).astype(float)
        logger.info("  Sex encoded: '%s' → 1, other → 0.", positive_value)
    return df


def apply_bounds(df: pd.DataFrame) -> pd.DataFrame:
    n_clipped = 0
    for col, (lo, hi) in BOUNDS.items():
        if col in df.columns:
            mask = df[col].notna() & ((df[col] < lo) | (df[col] > hi))
            n_clipped += int(mask.sum())
            df.loc[mask, col] = np.nan
    if n_clipped:
        logger.warning(
            "  Physiological bounds: set %d out-of-range values to NaN.", n_clipped
        )
    else:
        logger.info("  Physiological bounds: all values within range.")
    return df


def deduplicate(df: pd.DataFrame, patient_col: str = "patient_id",
                date_col: str = "visit_date") -> pd.DataFrame:
    """Keep first visit per patient (sorted by visit_date)."""
    before = len(df)
    df = df.sort_values(date_col)
    df = df.drop_duplicates(subset=[patient_col], keep="first")
    after  = len(df)
    logger.info(
        "  Deduplication: removed %d repeat visits (%d → %d unique patients).",
        before - after, before, after,
    )
    return df.reset_index(drop=True)


def filter_adults(df: pd.DataFrame, min_age: int = 18) -> pd.DataFrame:
    before = len(df)
    df = df[df["age"] >= min_age].copy()
    logger.info(
        "  Age filter (≥%d): removed %d minors (%d encounters retained).",
        min_age, before - len(df), len(df),
    )
    return df.reset_index(drop=True)


def validate_temporal_split(df: pd.DataFrame, cutoff: str, val_start: str) -> None:
    cutoff_dt    = pd.Timestamp(cutoff)
    val_start_dt = pd.Timestamp(val_start)
    train_n = (df["visit_date"] < cutoff_dt).sum()
    val_n   = (df["visit_date"] >= val_start_dt).sum()
    if train_n == 0:
        raise ValueError(
            f"No training encounters found before {cutoff}. "
            "Check visit_date column and training_cutoff_date in config.yaml."
        )
    if val_n == 0:
        raise ValueError(
            f"No validation encounters found on or after {val_start}. "
            "Check validation_start_date in config.yaml."
        )
    logger.info(
        "  Temporal split: %d training encounters (before %s), "
        "%d validation encounters (from %s).",
        train_n, cutoff, val_n, val_start,
    )


def print_summary(df: pd.DataFrame, cutoff: str) -> None:
    """Print cohort characteristics matching Table 1 of the paper."""
    cutoff_dt = pd.Timestamp(cutoff)
    train = df[df["visit_date"] < cutoff_dt]
    val   = df[df["visit_date"] >= cutoff_dt]

    logger.info("=" * 60)
    logger.info("COHORT SUMMARY")
    logger.info("=" * 60)
    logger.info("  Total encounters   : %d", len(df))
    logger.info("  Training set       : %d", len(train))
    logger.info("  Validation set     : %d", len(val))
    logger.info("  Admission rate     : %.1f%%", df["hospital_admission"].mean()*100)
    logger.info("  Median age (IQR)   : %.0f (%.0f–%.0f)",
                df["age"].median(),
                df["age"].quantile(0.25),
                df["age"].quantile(0.75))
    geri = df["age"] >= 65
    logger.info("  Geriatric (≥65)    : %d (%.1f%%)",
                geri.sum(), geri.mean()*100)
    logger.info("  Female             : %.1f%%",
                (df["sex"] == 0).mean()*100)
    logger.info("  Chief CC missing   : %.1f%%",
                df["chief_complaint"].isna().mean()*100)
    logger.info("  SpO2 missing       : %.1f%%",
                df["spo2"].isna().mean()*100 if "spo2" in df.columns else float("nan"))
    logger.info("=" * 60)


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    col_map   = cfg.get("column_mapping", {})
    sex_pos   = cfg.get("sex_positive_value", "Male")
    min_age   = cfg["data"].get("min_age", 18)
    cutoff    = cfg["data"]["training_cutoff_date"]
    val_start = cfg["data"]["validation_start_date"]

    # ── Load ────────────────────────────────────────────────────────────
    logger.info("Loading raw data from: %s", args.input)
    df = pd.read_csv(args.input, low_memory=False)
    logger.info("  Loaded %d rows × %d columns.", len(df), len(df.columns))

    # ── Validate raw columns ─────────────────────────────────────────────
    logger.info("Validating columns …")
    validate_columns(df, REQUIRED_COLS)

    # ── Apply column mapping ─────────────────────────────────────────────
    logger.info("Renaming columns …")
    df = apply_column_mapping(df, col_map)

    # ── Parse dates ──────────────────────────────────────────────────────
    df["visit_date"] = pd.to_datetime(df["visit_date"], errors="coerce")
    n_bad_dates = df["visit_date"].isna().sum()
    if n_bad_dates > 0:
        logger.warning("  %d rows have unparseable visit_date — dropped.", n_bad_dates)
        df = df.dropna(subset=["visit_date"])

    # ── Encode sex ───────────────────────────────────────────────────────
    logger.info("Encoding sex variable …")
    df = encode_sex(df, positive_value=sex_pos)

    # ── Physiological bounds ─────────────────────────────────────────────
    logger.info("Applying physiological bounds …")
    df = apply_bounds(df)

    # ── Deduplicate (first visit per patient) ────────────────────────────
    logger.info("Deduplicating (first visit per patient) …")
    df = deduplicate(df, patient_col="patient_id", date_col="visit_date")

    # ── Filter adults ────────────────────────────────────────────────────
    logger.info("Filtering to adults (age ≥ %d) …", min_age)
    df = filter_adults(df, min_age=min_age)

    # ── Validate temporal split ──────────────────────────────────────────
    logger.info("Validating temporal split …")
    validate_temporal_split(df, cutoff=cutoff, val_start=val_start)

    # ── Summary ──────────────────────────────────────────────────────────
    print_summary(df, cutoff=cutoff)

    # ── Write output ──────────────────────────────────────────────────────
    if args.dry_run:
        logger.info("Dry run: output not written.")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Cleaned dataset written → %s (%d rows)", out_path, len(df))


if __name__ == "__main__":
    main()
