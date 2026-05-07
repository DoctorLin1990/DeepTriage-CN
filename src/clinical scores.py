#!/usr/bin/env python3
"""
clinical_scores.py

Calculates traditional clinical early warning scores for benchmarking,
as described in Section 2.1 and Section 3.6.

Scores implemented:
    - ESI (Emergency Severity Index): Typically assigned by triage nurses.
      Since we lack resource prediction, we estimate ESI from vitals and age
      using a heuristic mapping based on the ESI algorithm v4. This is a
      simplified retrospective proxy, as noted in the paper (Section 3.6).
    - NEWS2 (National Early Warning Score 2): Aggregates weighted deviations
      in respiratory rate, oxygen saturation, temperature, systolic BP,
      heart rate, and level of consciousness.
    - MEWS (Modified Early Warning Score): A simpler 5-parameter version
      (no oxygen saturation scale in classic MEWS; we implement standard MEWS).

These scores are calculated retrospectively from triage vitals.
"""

import numpy as np
import pandas as pd
from typing import Tuple


def compute_esi_level(age: np.ndarray, hr: np.ndarray, rr: np.ndarray,
                      sbp: np.ndarray, spo2: np.ndarray, temp: np.ndarray) -> np.ndarray:
    """
    Estimate ESI level (1-5) from triage vitals.
    This is a heuristic proxy for the paper's analysis; true ESI requires
    nurse assessment of resource needs.

    Rules (simplified from ESI v4 handbook):
        Level 1: Immediate life-saving intervention (unresponsive, severe
                 respiratory distress, SpO2 < 90%, HR < 40 or > 130, SBP < 80)
        Level 2: High-risk situation (confused, severe pain, HR > 100, RR > 24,
                 SpO2 < 92%)
        Level 3: Expected to require multiple resources (stable vitals but
                 with comorbidity flags or abnormal vitals not life-threatening)
        Level 4: One resource needed
        Level 5: No resources needed

    Returns:
        Array of ESI levels (1-5, where 1 is most urgent).
    """
    esi = np.full_like(age, 3, dtype=int)  # Default to level 3 (most common)

    # Level 1: Life-threatening
    life_threat = (
        (spo2 < 90) |
        (hr < 40) | (hr > 130) |
        (respiratory_rate := rr) < 8 | (rr > 36) |
        (sbp < 80)
    )
    esi[life_threat] = 1

    # Level 2: High risk (but not life-threatening)
    high_risk = (
        (~life_threat) &
        ((spo2 < 92) | (hr > 100) | (rr > 24) |
         (sbp < 100) | (age >= 65))
    )
    esi[high_risk] = 2

    # The remainder stay at level 3 unless we have data suggesting level 4/5
    return esi


def compute_news2(
    rr: np.ndarray,
    spo2: np.ndarray,
    temp: np.ndarray,
    sbp: np.ndarray,
    hr: np.ndarray,
    consciousness_avpu: np.ndarray = None,
    on_oxygen: np.ndarray = None
) -> np.ndarray:
    """
    Calculate NEWS2 score.

    Components:
        - Respiration rate (0-3)
        - SpO₂ Scale 1 (0-3) or Scale 2 if hypercapnic risk (not simulated)
        - On supplemental oxygen? (0 or 2)
        - Temperature (0-3)
        - Systolic BP (0-3)
        - Heart rate (0-3)
        - Level of consciousness (AVPU) (0 or 3)

    Since we lack consciousness and oxygen therapy data, we assume:
        - AVPU = Alert (score 0) unless specified
        - Not on oxygen unless SpO2 < 94% (heuristic, Score 0 or 2)

    Args:
        rr: Respiratory rate (breaths/min)
        spo2: Peripheral oxygen saturation (%)
        temp: Temperature (°C)
        sbp: Systolic blood pressure (mmHg)
        hr: Heart rate (bpm)
        consciousness_avpu: Array with 'A', 'V', 'P', 'U'. If None, assume Alert.
        on_oxygen: Boolean array. If None, derived from spo2 < 94%.

    Returns:
        Integer NEWS2 total score (0-20+).
    """
    score = np.zeros_like(rr, dtype=float)

    # Respiration rate
    score += np.select([rr <= 8, rr <= 11, (rr >= 12) & (rr <= 20),
                        (rr >= 21) & (rr <= 24), rr >= 25],
                       [3, 1, 0, 2, 3], default=0)

    # SpO2 (Scale 1 – assuming no hypercapnic risk)
    score += np.select([spo2 <= 91, spo2 <= 93, spo2 <= 95, spo2 >= 96],
                       [3, 2, 1, 0], default=0)

    # Supplemental oxygen (approximated)
    if on_oxygen is None:
        on_oxygen = spo2 < 94  # heuristic
    score += np.where(on_oxygen, 2, 0)

    # Temperature
    score += np.select([temp <= 35.0, (temp > 35.0) & (temp <= 36.0),
                        (temp > 36.0) & (temp <= 38.0),
                        (temp > 38.0) & (temp <= 39.0), temp > 39.0],
                       [3, 1, 0, 1, 2], default=0)

    # Systolic BP
    score += np.select([sbp <= 90, sbp <= 100, sbp <= 110,
                        (sbp >= 111) & (sbp <= 219), sbp >= 220],
                       [3, 2, 1, 0, 3], default=0)

    # Heart rate
    score += np.select([hr <= 40, hr <= 50, (hr >= 51) & (hr <= 90),
                        (hr >= 91) & (hr <= 110), (hr >= 111) & (hr <= 130),
                        hr >= 131],
                       [3, 1, 0, 1, 2, 3], default=0)

    # Consciousness (if available, otherwise 0)
    if consciousness_avpu is not None:
        score += np.where(consciousness_avpu != "A", 3, 0)

    return score


def compute_mews(
    rr: np.ndarray,
    hr: np.ndarray,
    sbp: np.ndarray,
    temp: np.ndarray,
    consciousness_avpu: np.ndarray = None
) -> np.ndarray:
    """
    Calculate MEWS (Modified Early Warning Score).

    Components: RR, HR, SBP, Temperature, AVPU (each 0-3).
    SpO2 not included in classic MEWS.

    Returns:
        Integer MEWS score (0-15).
    """
    score = np.zeros_like(rr, dtype=float)

    # Respiration rate
    score += np.select([rr < 9, (rr >= 9) & (rr <= 14),
                        (rr >= 15) & (rr <= 20), (rr >= 21) & (rr <= 29),
                        rr >= 30],
                       [2, 0, 1, 2, 3], default=0)

    # Heart rate
    score += np.select([hr < 40, (hr >= 40) & (hr <= 50), (hr >= 51) & (hr <= 100),
                        (hr >= 101) & (hr <= 110), (hr >= 111) & (hr <= 129),
                        hr >= 130],
                       [3, 1, 0, 1, 2, 3], default=0)

    # Systolic BP
    score += np.select([sbp <= 70, (sbp >= 71) & (sbp <= 80),
                        (sbp >= 81) & (sbp <= 100), (sbp >= 101) & (sbp <= 199),
                        sbp >= 200],
                       [3, 2, 1, 0, 2], default=0)

    # Temperature
    score += np.select([temp < 35.0, (temp >= 35.0) & (temp <= 38.4),
                        temp >= 38.5],
                       [2, 0, 1], default=0)

    # AVPU (if available)
    if consciousness_avpu is not None:
        score += np.where(consciousness_avpu != "A", 3, 0)

    return score