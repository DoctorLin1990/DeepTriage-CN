#!/usr/bin/env python3
"""
generate_synthetic_data.py

Generates a synthetic dataset that statistically mirrors the ED triage cohort
described in the DeepTriage-CN paper (Section 4.1, Table 1).

Purpose:
    Since real patient data cannot be publicly shared due to IRB restrictions,
    this script produces synthetic data for testing, debugging, and verifying
    the computational reproducibility of the pipeline.

Output:
    data/sample_data.csv (10,000 rows, columns as documented in data/README.md)

Usage:
    python data/generate_synthetic_data.py
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import os
import sys

# ---------------------------------------------------------------------------
# 0. Configuration – anchored to the paper
# ---------------------------------------------------------------------------
N_TOTAL = 10_000
N_TRAIN = 8_000           # visits dated 2023-01-01 to 2023-12-31
N_VAL = 2_000             # visits dated 2024-01-01 to 2024-03-31
RANDOM_SEED = 42

# Cohort characteristics (Section 4.1)
ADMISSION_RATE_OVERALL = 0.220
ADMISSION_RATE_YOUNG = 0.190    # age < 65
ADMISSION_RATE_ELDER = 0.337    # age ≥ 65
PROPORTION_ELDER = 0.205        # 20.5% aged ≥ 65
MEDIAN_AGE = 48
AGE_IQR_LOW = 34
AGE_IQR_HIGH = 62
PROPORTION_FEMALE = 0.485

# Missingness proportions (Table 1)
MISSING_RATES = {
    "temperature": 0.035,
    "heart_rate": 0.021,
    "respiratory_rate": 0.052,
    "sbp": 0.018,
    "dbp": 0.020,
    "spo2": 0.082,
    "chief_complaint": 0.024,
}

# Chief complaint characteristics (Section 3.4)
MEAN_COMPLAINT_LENGTH = 8.3
SD_COMPLAINT_LENGTH = 4.1
MAX_TEXT_LENGTH = 64

# Clinical text templates – designed to mirror the semantic features
# identified by SHAP analysis (Figure 5)
COMPLAINT_TEMPLATES = {
    "chest_pain": [
        "胸痛伴胸闷", "胸痛三小时", "心前区疼痛", "胸痛并伴有呼吸急促",
        "左侧胸痛", "胸骨后压榨性疼痛", "胸痛伴出汗",
    ],
    "dyspnea": [
        "呼吸困难", "气喘加重", "呼吸急促两天", "活动后气促",
        "夜间阵发性呼吸困难", "喘息伴咳嗽",
    ],
    "altered_mental": [
        "精神差反应迟钝", "意识模糊", "嗜睡状态", "言语不清",
        "突发意识丧失", "昏厥后清醒",
    ],
    "dizziness": [
        "头晕", "头晕乏力", "头重脚轻", "眩晕伴恶心",
    ],
    "fever": [
        "发热两天", "发热咳嗽", "高热不退", "发热寒战",
        "低热乏力", "发热伴咽痛",
    ],
    "abdominal": [
        "腹痛", "上腹痛", "右下腹痛", "腹痛恶心",
        "腹胀不适", "腹泻呕吐",
    ],
    "trauma": [
        "摔倒致右髋疼痛", "交通事故后颈部疼痛", "高处坠落伤",
        "头部外伤", "扭伤左踝",
    ],
    "other": [
        "全身乏力", "食欲不振一周", "失眠多梦", "心悸",
        "双下肢水肿", "皮肤黄染", "尿频尿急", "关节疼痛",
    ],
}

# Category weights approximating ED prevalence
COMPLAINT_CATEGORY_WEIGHTS = {
    "chest_pain": 0.12,
    "dyspnea": 0.10,
    "altered_mental": 0.06,
    "dizziness": 0.10,
    "fever": 0.18,
    "abdominal": 0.14,
    "trauma": 0.12,
    "other": 0.18,
}

np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. Helper functions
# ---------------------------------------------------------------------------
def generate_age(n: int, prop_elder: float) -> np.ndarray:
    """
    Generate age values matching the reported distribution.
    Uses a log-normal approximation for adult ages (≥18) with a bump at 65+.
    """
    ages = np.zeros(n)
    n_elder = int(n * prop_elder)
    n_young = n - n_elder

    # Young group: 18 to 64
    young = np.random.triangular(18, MEDIAN_AGE, 64, size=n_young).astype(int)

    # Elder group: 65 to ~95
    elder = np.random.triangular(65, 74, 95, size=n_elder).astype(int)

    ages[:n_young] = young
    ages[n_young:] = elder
    np.random.shuffle(ages)
    return ages


def generate_vital_signs(n: int, admission_mask: np.ndarray,
                         ages: np.ndarray) -> pd.DataFrame:
    """
    Generate synthetic vital signs with distributions that differ between
    admitted and discharged patients. Older admitted patients tend to have
    more extreme values (mirroring the paper's clinical narrative).
    """
    # Baseline means for healthy physiology
    mu_temp = 36.8
    mu_hr = 78
    mu_rr = 16
    mu_sbp = 125
    mu_dbp = 78
    mu_spo2 = 98

    # Admitted patients receive higher variance and shifted means
    adm_factor = np.where(admission_mask == 1,
                          np.random.uniform(1.0, 2.5, size=n),
                          1.0)
    age_factor = np.where(ages >= 65, 1.15, 1.0)

    temp = np.random.normal(mu_temp, 0.8 * adm_factor * age_factor, n)
    hr = np.random.normal(mu_hr, 15 * adm_factor, n).astype(int)
    rr = np.random.normal(mu_rr, 4 * adm_factor, n).astype(int)
    sbp = np.random.normal(mu_sbp, 20 * adm_factor * age_factor, n).astype(int)
    dbp = np.random.normal(mu_dbp, 12 * adm_factor, n).astype(int)
    spo2 = np.random.normal(mu_spo2, 3 * adm_factor, n)

    # Clamp to plausible physiological ranges
    temp = np.clip(temp, 34.0, 42.0)
    hr = np.clip(hr, 30, 200)
    rr = np.clip(rr, 8, 50)
    sbp = np.clip(sbp, 60, 250)
    dbp = np.clip(dbp, 30, 150)
    spo2 = np.clip(spo2, 60, 100)

    return pd.DataFrame({
        "temperature": np.round(temp, 1),
        "heart_rate": hr,
        "respiratory_rate": rr,
        "sbp": sbp,
        "dbp": dbp,
        "spo2": spo2,
    })


def generate_complaint(n: int, adm_mask: np.ndarray) -> list:
    """Generate synthetic chief complaints in Mandarin Chinese."""
    categories = list(COMPLAINT_CATEGORY_WEIGHTS.keys())
    weights = list(COMPLAINT_CATEGORY_WEIGHTS.values())
    chosen_categories = np.random.choice(categories, size=n, p=weights)

    complaints = []
    for i, cat in enumerate(chosen_categories):
        base_text = np.random.choice(COMPLAINT_TEMPLATES[cat])
        # Add modifiers or additional words to achieve target length distribution
        target_len = max(2, int(np.random.normal(MEAN_COMPLAINT_LENGTH, SD_COMPLAINT_LENGTH)))
        # Approximately match Chinese token count
        if len(base_text) < target_len:
            modifiers = ["加重", "持续", "反复", "本次", "来院就诊", "三天", "一周"]
            add_count = min(len(modifiers), target_len - len(base_text))
            for j in range(add_count):
                base_text += np.random.choice(modifiers)
        complaints.append(base_text[:MAX_TEXT_LENGTH])
    return complaints


def apply_mcar_missing(df: pd.DataFrame, rates: dict) -> pd.DataFrame:
    """Apply MCAR (Missing Completely At Random) missingness to baseline data."""
    for col, rate in rates.items():
        if col in df.columns:
            mask = np.random.random(len(df)) < rate
            df.loc[mask, col] = np.nan
    return df


# ---------------------------------------------------------------------------
# 2. Main generation pipeline
# ---------------------------------------------------------------------------
def main():
    print("Generating synthetic ED triage dataset (N = 10,000)...")

    # --- Step 1: Demographics ---
    ages = generate_age(N_TOTAL, PROPORTION_ELDER)
    sex = np.random.binomial(1, 1.0 - PROPORTION_FEMALE, N_TOTAL)

    is_elder = ages >= 65

    # --- Step 2: Admission outcome ---
    adm_prob = np.where(is_elder, ADMISSION_RATE_ELDER, ADMISSION_RATE_YOUNG)
    # Add age-dependent risk to get nuanced probabilities
    adm_prob += 0.02 * (ages > 75).astype(float)
    adm_prob += 0.01 * (ages > 85).astype(float)
    adm_prob = np.clip(adm_prob, 0.05, 0.90)

    admission = np.random.binomial(1, adm_prob, N_TOTAL)

    # --- Step 3: Vital signs ---
    vitals = generate_vital_signs(N_TOTAL, admission, ages)

    # --- Step 4: Chief complaints ---
    complaints = generate_complaint(N_TOTAL, admission)

    # --- Step 5: Assemble DataFrame ---
    df = pd.DataFrame({
        "visit_id": [f"ED{2023000000 + i}" for i in range(N_TOTAL)],
        "age": ages,
        "sex": sex,
    })
    df = pd.concat([df, vitals], axis=1)
    df["chief_complaint"] = complaints
    df["hospital_admission"] = admission

    # --- Step 6: Apply missingness (Table 1 rates) ---
    vitals_cols = ["temperature", "heart_rate", "respiratory_rate",
                   "sbp", "dbp", "spo2"]
    missing_rates_for_vitals = {k: v for k, v in MISSING_RATES.items()
                                if k in vitals_cols}
    df = apply_mcar_missing(df, missing_rates_for_vitals)

    # Apply chief complaint missingness
    cc_mask = np.random.random(N_TOTAL) < MISSING_RATES["chief_complaint"]
    df.loc[cc_mask, "chief_complaint"] = ""

    # --- Step 7: Assign dates and split into train/val ---
    # Training: 2023-01-01 to 2023-12-31
    train_start = datetime(2023, 1, 1)
    train_dates = [train_start + timedelta(days=int(x))
                   for x in np.random.randint(0, 364, N_TRAIN)]

    # Validation: 2024-01-01 to 2024-03-31
    val_start = datetime(2024, 1, 1)
    val_dates = [val_start + timedelta(days=int(x))
                 for x in np.random.randint(0, 90, N_VAL)]

    all_dates = train_dates + val_dates
    df["visit_date"] = all_dates
    df["visit_date"] = pd.to_datetime(df["visit_date"])

    # --- Step 8: Save ---
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "sample_data.csv")
    df.to_csv(output_path, index=False, encoding="utf-8-sig")

    # --- Step 9: Print summary statistics for verification ---
    print(f"\nSynthetic dataset saved to: {output_path}")
    print(f"Total encounters: {len(df)}")
    print(f"Training encounters (2023): {N_TRAIN}")
    print(f"Validation encounters (Jan–Mar 2024): {N_VAL}")
    print(f"Overall admission rate: {df['hospital_admission'].mean():.3f}")
    print(f"  Age < 65:  {df[df.age < 65]['hospital_admission'].mean():.3f}")
    print(f"  Age ≥ 65:  {df[df.age >= 65]['hospital_admission'].mean():.3f}")
    print(f"Proportion aged ≥65: {is_elder.mean():.3f}")
    print(f"Proportion female: {1.0 - df.sex.mean():.3f}")
    print(f"Chief complaint missing rate: {(df.chief_complaint == '').mean():.3f}")
    for col in vitals_cols:
        print(f"  {col} missing: {df[col].isna().mean():.3f}")
    print("\nDone.")


if __name__ == "__main__":
    main()