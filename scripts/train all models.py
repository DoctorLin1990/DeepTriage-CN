#!/usr/bin/env python3
"""
train_all_models.py

Trains all models described in the paper and saves them to outputs/models/.

Models:
    1. DeepTriage-CN (multimodal late-fusion)
    2. TabNet (structured-only deep learning)
    3. Vitals-only XGBoost
    4. Random Forest
    5. Text-only Logistic Regression

Also computes clinical scores (NEWS2, MEWS, ESI) on the validation set.

Usage:
    python scripts/train_all_models.py --config config.yaml
"""

import argparse
import logging
import yaml
import sys
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Ensure project root is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    preprocess_structured,
    preprocess_text,
    get_labels,
)
from src.text_encoder import TextEncoder
from src.fusion_model import DeepTriageCN
from src.tabnet_model import TabNetWrapper
from src.baseline_models import (
    VitalsOnlyXGBoost,
    RandomForestBaseline,
    TextOnlyLogisticRegression,
)
from src.clinical_scores import compute_news2, compute_mews, compute_esi_level

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config["data"]
    xgb_cfg = config["xgboost"]
    tabnet_cfg = config["tabnet"]
    text_cfg = config["text_processing"]
    output_dir = Path(config["output_paths"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    random_state = config["random_state"]

    # ---- Load and preprocess ----
    train_df, val_df = load_and_split_data(
        data_cfg["raw_data_path"],
        training_cutoff_date=data_cfg["training_cutoff_date"],
        validation_start_date=data_cfg["validation_start_date"],
        random_state=random_state,
    )
    X_train_raw, X_val_raw, scaler = preprocess_structured(train_df, val_df)
    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    # Calculate scale_pos_weight for imbalanced data
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # ---- 1. DeepTriage-CN ----
    logger.info("Training DeepTriage-CN ...")
    text_encoder = TextEncoder(
        model_name=text_cfg["model_name"],
        max_length=text_cfg["max_length"],
    )
    deeptriage = DeepTriageCN(
        text_encoder=text_encoder,
        random_state=random_state,
    )
    deeptriage.fit(
        train_texts.tolist(), X_train_raw, y_train,
        scale_pos_weight=scale_pos_weight,
    )
    deeptriage.save(str(output_dir / "deeptriage_cn.pkl"))

    # ---- 2. TabNet ----
    logger.info("Training TabNet ...")
    tabnet = TabNetWrapper(
        n_d=tabnet_cfg["n_d"],
        n_a=tabnet_cfg["n_a"],
        n_steps=tabnet_cfg["n_steps"],
        gamma=tabnet_cfg["gamma"],
        lambda_sparse=tabnet_cfg["lambda_sparse"],
        seed=random_state,
    )
    eval_set = [(X_val_raw, y_val)]
    tabnet.fit(
        X_train_raw, y_train,
        eval_set=eval_set,
        max_epochs=tabnet_cfg["max_epochs"],
        batch_size=tabnet_cfg["batch_size"],
        virtual_batch_size=tabnet_cfg["virtual_batch_size"],
    )
    tabnet.save(str(output_dir / "tabnet"))

    # ---- 3. Vitals-only XGBoost ----
    logger.info("Training Vitals-only XGBoost ...")
    vit_xgb = VitalsOnlyXGBoost(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=random_state,
    )
    vit_xgb.fit(X_train_raw, y_train)
    joblib.dump(vit_xgb, output_dir / "vitals_only_xgb.pkl")

    # ---- 4. Random Forest ----
    logger.info("Training Random Forest ...")
    rf = RandomForestBaseline(n_estimators=200, max_depth=6, random_state=random_state)
    rf.fit(X_train_raw, y_train)
    joblib.dump(rf, output_dir / "random_forest.pkl")

    # ---- 5. Text-only Logistic Regression ----
    logger.info("Training Text-only Logistic Regression ...")
    train_emb = text_encoder.encode(train_texts.tolist())
    text_lr = TextOnlyLogisticRegression(random_state=random_state)
    text_lr.fit(train_emb, y_train)
    joblib.dump(text_lr, output_dir / "text_only_lr.pkl")

    # ---- Clinical scores (validation set) ----
    logger.info("Computing clinical scores on validation set ...")
    val_scores = pd.DataFrame({
        "NEWS2": compute_news2(
            val_df["respiratory_rate"].values,
            val_df["spo2"].values,
            val_df["temperature"].values,
            val_df["sbp"].values,
            val_df["heart_rate"].values,
        ),
        "MEWS": compute_mews(
            val_df["respiratory_rate"].values,
            val_df["heart_rate"].values,
            val_df["sbp"].values,
            val_df["temperature"].values,
        ),
        "ESI": compute_esi_level(
            val_df["age"].values,
            val_df["heart_rate"].values,
            val_df["respiratory_rate"].values,
            val_df["sbp"].values,
            val_df["spo2"].values,
            val_df["temperature"].values,
        ),
    })
    val_scores.to_csv(output_dir / "clinical_scores_val.csv", index=False)

    logger.info("All models trained and saved successfully.")


if __name__ == "__main__":
    main()