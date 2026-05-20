#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generates a synthetic dataset that mimics the statistical properties of the
real cohort described in Section 3.2 / Table 1, enabling smoke-tests and
CI runs without exposing protected health information.

Synthetic dataset properties (aligned with paper cohort):
    Total encounters : 10,000
    Training period  : 2023-01-01 – 2023-12-31 (8,000 visits)
    Validation period: 2024-01-01 – 2024-03-31 (2,000 visits)
    Admission rate   : 22.0%
    Median age       : 48 years (IQR 34–62)
    Geriatric (≥65)  : 20.5% of cohort; admission rate 33.7%
    Missing rates    : match Table 1 (spo2 8.2%, rr 5.2%, temp 3.5%, …)
    Chief complaint  : Simplified Mandarin text drawn from a fixed vocabulary

IMPORTANT: This dataset is entirely synthetic. It replicates aggregate
statistics but does NOT contain any real patient information. Results
produced from this dataset will differ from the paper's reported values.

Usage:
    python data/generate_synthetic_data.py
    # Writes: data/synthetic_ed_visits.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Random seed for reproducibility
RNG = np.random.default_rng(42)

# Synthetic chief complaint vocabulary (Section 3.4: mean 8.3 words, SD 4.1)
_COMPLAINTS_YOUNG = [
    "腹痛 恶心 呕吐 发热",
    "头痛 头晕 乏力",
    "胸痛 心悸 气短",
    "外伤 摔伤 右腿疼痛",
    "发热 咳嗽 咽痛",
    "腰痛 下肢麻木",
    "腹泻 呕吐 脱水",
    "过敏反应 皮疹 瘙痒",
    "尿频 尿痛 血尿",
    "心慌 胸闷 出汗",
]
_COMPLAINTS_ELDER = [
    "头晕 乏力 行走不稳",
    "胸闷 气促 端坐呼吸",
    "意识模糊 嗜睡 不能进食",
    "发热 咳嗽 咳痰",
    "腹痛 纳差 便秘",
    "头痛 视物模糊 血压升高",
    "跌倒 髋部疼痛 活动受限",
    "心悸 气短",
    "全身乏力",        # sparse (3 words)
    "头晕",            # sparse (1 word)
]


def _make_vitals(n: int, is_admitted: np.ndarray) -> dict:
    """Generate synthetic vital signs with realistic distributions."""
    admitted = is_admitted.astype(bool)
    normal   = ~admitted

    # Temperature (°C) — admitted patients slightly higher
    temp = np.where(
        admitted,
        RNG.normal(37.8, 0.8, n),
        RNG.normal(37.0, 0.5, n),
    )
    temp = np.clip(temp, 35.0, 41.0)

    # Heart rate (bpm)
    hr = np.where(
        admitted,
        RNG.normal(95, 18, n),
        RNG.normal(78, 12, n),
    )
    hr = np.clip(hr, 40, 160)

    # Respiratory rate (breaths/min)
    rr = np.where(
        admitted,
        RNG.normal(22, 5, n),
        RNG.normal(16, 3, n),
    )
    rr = np.clip(rr, 8, 40)

    # SBP (mmHg)
    sbp = np.where(
        admitted,
        RNG.normal(118, 22, n),
        RNG.normal(128, 16, n),
    )
    sbp = np.clip(sbp, 70, 220)

    # DBP (mmHg)
    dbp = sbp / RNG.uniform(1.5, 2.0, n)
    dbp = np.clip(dbp, 40, 130)

    # SpO2 (%)
    spo2 = np.where(
        admitted,
        RNG.normal(95.5, 3.5, n),
        RNG.normal(98.5, 1.2, n),
    )
    spo2 = np.clip(spo2, 70, 100)

    return {
        "temperature":      temp,
        "heart_rate":       hr,
        "respiratory_rate": rr,
        "sbp":              sbp,
        "dbp":              dbp,
        "spo2":             spo2,
    }


def _apply_missing(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply per-feature missingness aligned with Table 1 of the paper.
    """
    missing_rates = {
        "temperature":      0.035,
        "heart_rate":       0.021,
        "respiratory_rate": 0.052,
        "sbp":              0.018,
        "dbp":              0.020,
        "spo2":             0.082,
        "chief_complaint":  0.024,
    }
    for col, rate in missing_rates.items():
        if col not in df.columns:
            continue
        mask = RNG.random(len(df)) < rate
        df.loc[mask, col] = np.nan
    return df


def generate_synthetic_cohort(n_total: int = 10_000) -> pd.DataFrame:
    """
    Generate the full synthetic cohort.

    Args:
        n_total : Total number of encounters.

    Returns:
        DataFrame with columns matching the real data schema.
    """
    # --- Temporal split ---
    n_train = 8_000
    n_val   = n_total - n_train

    # Training dates (2023)
    train_dates = pd.date_range("2023-01-01", "2023-12-31", periods=n_train)
    val_dates   = pd.date_range("2024-01-01", "2024-03-31",  periods=n_val)
    all_dates   = np.concatenate([train_dates, val_dates])

    # --- Age (median 48, IQR 34–62; geriatric ≥65 → 20.5%) ---
    age = RNG.gamma(shape=4.0, scale=12.0, size=n_total) + 18
    # Force ~20.5% geriatric
    n_elder = int(n_total * 0.205)
    elder_ages = RNG.uniform(65, 95, n_elder)
    elder_idx  = RNG.choice(n_total, size=n_elder, replace=False)
    age[elder_idx] = elder_ages
    age = np.clip(age, 18, 99).round(0).astype(int)

    is_elder = age >= 65

    # --- Sex (48.5% female = 0) ---
    sex = RNG.binomial(1, 0.515, n_total)  # 1 = male

    # --- Admission label ---
    # Baseline admission rate 22%; geriatric 33.7%
    admission_prob = np.where(is_elder, 0.337, 0.190)
    hospital_admission = RNG.binomial(1, admission_prob, n_total)

    # --- Vital signs ---
    vitals = _make_vitals(n_total, hospital_admission)

    # --- Chief complaints ---
    complaints = []
    for i in range(n_total):
        pool = _COMPLAINTS_ELDER if is_elder[i] else _COMPLAINTS_YOUNG
        complaints.append(RNG.choice(pool))

    # --- Patient IDs (for deduplication demo) ---
    patient_ids = np.arange(1, n_total + 1)

    df = pd.DataFrame({
        "patient_id":        patient_ids,
        "visit_date":        all_dates,
        "age":               age,
        "sex":               sex,
        "chief_complaint":   complaints,
        "hospital_admission": hospital_admission,
        **vitals,
    })

    # Apply realistic missingness
    df = _apply_missing(df)

    logger.info(
        f"Generated {len(df)} encounters | "
        f"admission rate {df['hospital_admission'].mean():.3f} | "
        f"geriatric {is_elder.mean():.3f}"
    )
    return df


def main() -> None:
    out_path = Path(__file__).parent / "synthetic_ed_visits.csv"
    df = generate_synthetic_cohort(n_total=10_000)
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Synthetic dataset saved → {out_path}")
    logger.info(
        "\nNOTE: This file is synthetic. Results will differ from the paper.\n"
        "The real dataset cannot be shared due to institutional privacy "
        "regulations (see Data Availability Statement)."
    )


if __name__ == "__main__":
    main()
