#!/usr/bin/env python3
"""
clinical_scores.py

Calculates traditional clinical early warning scores for benchmarking,
as described in Section 2.1 and Section 3.6.

Scores implemented:
    - ESI (Emergency Severity Index): Estimated from vitals via a heuristic
      mapping based on the ESI algorithm v4 (simplified retrospective proxy,
      as noted in Section 3.6).
    - NEWS2 (National Early Warning Score 2): Aggregates weighted deviations
      in respiratory rate, oxygen saturation, temperature, systolic BP,
      heart rate, and level of consciousness.
    - MEWS (Modified Early Warning Score): Classic 5-parameter version.

Bug fixed (B1): The original compute_esi_level contained an illegal walrus
operator assignment inside a bitwise expression:
    (respiratory_rate := rr) < 8 | (rr > 36)
This caused a SyntaxError and also incorrect operator precedence because
  `<` binds before `|`, making the expression evaluate as:
    ((respiratory_rate := rr) < (8 | (rr > 36)))
  which is semantically wrong.  The corrected version uses simple variables.
"""

import numpy as np
from typing import Optional


def compute_esi_level(
    age: np.ndarray,
    hr: np.ndarray,
    rr: np.ndarray,
    sbp: np.ndarray,
    spo2: np.ndarray,
    temp: np.ndarray,
) -> np.ndarray:
    """
    Estimate ESI level (1-5) from triage vitals.

    This is a heuristic proxy for the paper's analysis; true ESI requires
    nurse assessment of resource needs (Section 3.6).

    Rules (simplified from ESI v4 handbook):
        Level 1: Immediate life-saving intervention required
                 (SpO2 < 90, HR < 40 or > 130, RR < 8 or > 36, SBP < 80)
        Level 2: High-risk situation
                 (SpO2 < 92, HR > 100, RR > 24, SBP < 100, or age ≥ 65)
        Level 3: Multiple resources expected (default)

    Args:
        age:  Patient ages (years).
        hr:   Heart rate (bpm).
        rr:   Respiratory rate (breaths/min).
        sbp:  Systolic blood pressure (mmHg).
        spo2: Peripheral oxygen saturation (%).
        temp: Body temperature (°C).  (Reserved for future ESI extensions.)

    Returns:
        np.ndarray of int (1–5), lower = more urgent.
    """
    # Convert inputs to float arrays to allow NaN-tolerant comparisons
    age  = np.asarray(age,  dtype=float)
    hr   = np.asarray(hr,   dtype=float)
    rr   = np.asarray(rr,   dtype=float)
    sbp  = np.asarray(sbp,  dtype=float)
    spo2 = np.asarray(spo2, dtype=float)

    # Default level 3 (most common triage category)
    esi = np.full(age.shape, 3, dtype=int)

    # --- Level 1: Life-threatening ---
    # NaN comparisons return False, so missing values do not trigger level 1
    life_threat = (
        (spo2 < 90)
        | (hr < 40)
        | (hr > 130)
        | (rr < 8)          # ← fixed: separate conditions (not walrus)
        | (rr > 36)         # ← fixed
        | (sbp < 80)
    )
    esi[life_threat] = 1

    # --- Level 2: High risk (not immediately life-threatening) ---
    high_risk = (
        (~life_threat)
        & (
            (spo2 < 92)
            | (hr > 100)
            | (rr > 24)
            | (sbp < 100)
            | (age >= 65)
        )
    )
    esi[high_risk] = 2

    # Levels 4 and 5 cannot be reliably inferred from vitals alone
    # (they require resource estimation); all remaining cases stay at 3.
    return esi


def compute_news2(
    rr: np.ndarray,
    spo2: np.ndarray,
    temp: np.ndarray,
    sbp: np.ndarray,
    hr: np.ndarray,
    consciousness_avpu: Optional[np.ndarray] = None,
    on_oxygen: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate NEWS2 score (National Early Warning Score 2).

    Components and point values follow the Royal College of Physicians
    NEWS2 chart (2017 revision).  Missing vitals (NaN) contribute 0 points
    (conservative assumption).

    Args:
        rr:                 Respiratory rate (breaths/min).
        spo2:               Peripheral oxygen saturation (%).
        temp:               Temperature (°C).
        sbp:                Systolic blood pressure (mmHg).
        hr:                 Heart rate (bpm).
        consciousness_avpu: Optional array with values 'A','V','P','U'.
                            If None, assumes Alert (score 0).
        on_oxygen:          Optional boolean array for supplemental O2.
                            If None, derived heuristically from spo2 < 94%.

    Returns:
        np.ndarray of float NEWS2 total scores (0–20+).
    """
    rr   = np.asarray(rr,   dtype=float)
    spo2 = np.asarray(spo2, dtype=float)
    temp = np.asarray(temp, dtype=float)
    sbp  = np.asarray(sbp,  dtype=float)
    hr   = np.asarray(hr,   dtype=float)

    score = np.zeros_like(rr, dtype=float)

    # Respiration rate (breaths/min)
    score += np.select(
        [rr <= 8, rr <= 11, (rr >= 12) & (rr <= 20), (rr >= 21) & (rr <= 24), rr >= 25],
        [3,       1,         0,                         2,                        3],
        default=0,
    )

    # SpO2 Scale 1 (no hypercapnic risk assumed)
    score += np.select(
        [spo2 <= 91, spo2 <= 93, spo2 <= 95, spo2 >= 96],
        [3,          2,          1,           0],
        default=0,
    )

    # Supplemental oxygen (heuristic: SpO2 < 94 %)
    if on_oxygen is None:
        on_oxygen = spo2 < 94
    score += np.where(on_oxygen, 2, 0)

    # Temperature (°C)
    score += np.select(
        [temp <= 35.0,
         (temp > 35.0) & (temp <= 36.0),
         (temp > 36.0) & (temp <= 38.0),
         (temp > 38.0) & (temp <= 39.0),
         temp > 39.0],
        [3, 1, 0, 1, 2],
        default=0,
    )

    # Systolic BP (mmHg)
    score += np.select(
        [sbp <= 90, sbp <= 100, sbp <= 110, (sbp >= 111) & (sbp <= 219), sbp >= 220],
        [3,         2,          1,           0,                             3],
        default=0,
    )

    # Heart rate (bpm)
    score += np.select(
        [hr <= 40,
         hr <= 50,
         (hr >= 51) & (hr <= 90),
         (hr >= 91) & (hr <= 110),
         (hr >= 111) & (hr <= 130),
         hr >= 131],
        [3, 1, 0, 1, 2, 3],
        default=0,
    )

    # AVPU consciousness (if available)
    if consciousness_avpu is not None:
        score += np.where(np.asarray(consciousness_avpu) != "A", 3, 0)

    return score


def compute_mews(
    rr: np.ndarray,
    hr: np.ndarray,
    sbp: np.ndarray,
    temp: np.ndarray,
    consciousness_avpu: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Calculate MEWS (Modified Early Warning Score).

    Classic 5-component version (RR, HR, SBP, Temperature, AVPU).
    SpO2 is not part of classic MEWS.

    Args:
        rr:                 Respiratory rate (breaths/min).
        hr:                 Heart rate (bpm).
        sbp:                Systolic blood pressure (mmHg).
        temp:               Body temperature (°C).
        consciousness_avpu: Optional AVPU array. None → assumes Alert.

    Returns:
        np.ndarray of float MEWS total scores (0–15).
    """
    rr   = np.asarray(rr,   dtype=float)
    hr   = np.asarray(hr,   dtype=float)
    sbp  = np.asarray(sbp,  dtype=float)
    temp = np.asarray(temp, dtype=float)

    score = np.zeros_like(rr, dtype=float)

    # Respiration rate
    score += np.select(
        [rr < 9, (rr >= 9) & (rr <= 14), (rr >= 15) & (rr <= 20),
         (rr >= 21) & (rr <= 29), rr >= 30],
        [2, 0, 1, 2, 3],
        default=0,
    )

    # Heart rate
    score += np.select(
        [hr < 40, (hr >= 40) & (hr <= 50), (hr >= 51) & (hr <= 100),
         (hr >= 101) & (hr <= 110), (hr >= 111) & (hr <= 129), hr >= 130],
        [3, 1, 0, 1, 2, 3],
        default=0,
    )

    # Systolic BP
    score += np.select(
        [sbp <= 70, (sbp >= 71) & (sbp <= 80), (sbp >= 81) & (sbp <= 100),
         (sbp >= 101) & (sbp <= 199), sbp >= 200],
        [3, 2, 1, 0, 2],
        default=0,
    )

    # Temperature
    score += np.select(
        [temp < 35.0, (temp >= 35.0) & (temp <= 38.4), temp >= 38.5],
        [2, 0, 1],
        default=0,
    )

    # AVPU (if provided)
    if consciousness_avpu is not None:
        score += np.where(np.asarray(consciousness_avpu) != "A", 3, 0)

    return score
