#!/usr/bin/env python3
"""
predict_single_patient.py

Demonstration script for running inference on a single synthetic patient.
Takes chief complaint and vital signs as command-line arguments and prints
the predicted probability of hospital admission using DeepTriage-CN.

Usage:
    python scripts/predict_single_patient.py \\
        --config config.yaml \\
        --complaint "胸痛并伴有呼吸急促" \\
        --age 75 --heart_rate 110 --respiratory_rate 26 \\
        --spo2 93 --temperature 37.0 --sbp 145 --dbp 88 --sex 1
"""

import argparse
import yaml
import sys
import numpy as np
import joblib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.fusion_model import DeepTriageCN
from src.text_encoder import TextEncoder
from src.structured_encoder import STRUCTURED_FEATURE_NAMES


def main():
    parser = argparse.ArgumentParser(
        description="Predict hospital admission for a single patient."
    )
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--complaint", required=True, help="Chief complaint text in Chinese")
    parser.add_argument("--age", type=float, required=True)
    parser.add_argument("--sex", type=int, required=True, choices=[0, 1],
                        help="0: female, 1: male")
    parser.add_argument("--temperature", type=float, default=37.0)
    parser.add_argument("--heart_rate", type=float, default=80)
    parser.add_argument("--respiratory_rate", type=float, default=16)
    parser.add_argument("--sbp", type=float, default=120)
    parser.add_argument("--dbp", type=float, default=80)
    parser.add_argument("--spo2", type=float, default=98)
    parser.add_argument("--model_path", default="outputs/models/deeptriage_cn.pkl",
                        help="Path to trained model")
    args = parser.parse_args()

    # Load config and model
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    model_path = Path(args.model_path)
    if not model_path.exists():
        print("Model not found. Run train_all_models.py first.")
        sys.exit(1)

    deeptriage = joblib.load(model_path)

    # Prepare inputs
    complaint = args.complaint
    structured = np.array([[
        args.age, args.sex, args.temperature, args.heart_rate,
        args.respiratory_rate, args.sbp, args.dbp, args.spo2,
    ]])

    proba = deeptriage.predict_proba([complaint], structured)[0]
    threshold = config["evaluation"]["optimal_threshold"]
    prediction = "ADMIT" if proba >= threshold else "DISCHARGE"

    print("\n--- DeepTriage-CN Prediction ---")
    print(f"Chief complaint: {complaint}")
    print(f"Age: {args.age}, Sex: {'Male' if args.sex==1 else 'Female'}")
    print(f"Vitals: Temp={args.temperature}, HR={args.heart_rate}, "
          f"RR={args.respiratory_rate}, SBP={args.sbp}, DBP={args.dbp}, SpO2={args.spo2}")
    print(f"\nPredicted admission probability: {proba:.3f}")
    print(f"Threshold (Youden-optimal): {threshold:.2f}")
    print(f"Decision: {prediction}")
    print("-" * 35)


if __name__ == "__main__":
    main()