#!/usr/bin/env python3
"""
train_all_models.py

Trains all models described in the paper and saves them to outputs/models/.

Models trained:
    1. DeepTriage-CN  – multimodal late-fusion (BERT + XGBoost)
    2. TabNet         – structured-only deep learning baseline
    3. Vitals-only XGBoost
    4. Random Forest
    5. Text-only Logistic Regression (BERT + LR)

Also computes clinical scores (NEWS2, MEWS, ESI) on the validation set.

Bug fixes applied:
    L1 / L2 – The original script called preprocess_structured() which returns
               standardised arrays, then passed those arrays to DeepTriageCN.fit()
               and TabNetWrapper.fit(), both of which perform their own internal
               standardisation → double standardisation, severely distorting
               feature distributions.

               Fix: DeepTriageCN receives *raw* arrays (via get_raw_structured()).
                    TabNet also expects raw input (it applies BN internally).
                    Only VitalsOnlyXGBoost, RandomForest, and TextOnlyLR receive
                    the pre-standardised arrays from preprocess_structured().

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

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import (
    load_and_split_data,
    get_raw_structured,
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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Train all DeepTriage-CN models.")
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML.")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    data_cfg   = config["data"]
    xgb_cfg    = config["xgboost"]
    tabnet_cfg = config["tabnet"]
    text_cfg   = config["text_processing"]
    output_dir = Path(config["output_paths"]["models_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    random_state = config["random_state"]

    # ------------------------------------------------------------------
    # 1. Load data and temporal split
    # ------------------------------------------------------------------
    logger.info("Loading and splitting data …")
    train_df, val_df = load_and_split_data(
        data_cfg["raw_data_path"],
        training_cutoff_date=data_cfg["training_cutoff_date"],
        validation_start_date=data_cfg["validation_start_date"],
        random_state=random_state,
    )

    # Raw (unstandardised) arrays — for DeepTriageCN and TabNet
    X_train_raw, X_val_raw = get_raw_structured(train_df, val_df)

    # Standardised arrays — for VitalsXGB, RandomForest, TextLR
    X_train_std, X_val_std, _ = preprocess_structured(train_df, val_df)

    train_texts, val_texts = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    n_neg = int(np.sum(y_train == 0))
    n_pos = int(np.sum(y_train == 1))
    scale_pos_weight = n_neg / n_pos if n_pos > 0 else 1.0
    logger.info(
        f"Class distribution — neg: {n_neg}, pos: {n_pos}, "
        f"scale_pos_weight: {scale_pos_weight:.3f}"
    )

    # ------------------------------------------------------------------
    # 2. DeepTriage-CN  (receives raw arrays; standardises internally)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training DeepTriage-CN …")
    text_encoder = TextEncoder(
        model_name=text_cfg["model_name"],
        max_length=text_cfg["max_length"],
    )
    deeptriage = DeepTriageCN(
        text_encoder=text_encoder,
        random_state=random_state,
    )
    deeptriage.fit(
        train_texts.tolist(),
        X_train_raw,          # ← raw, not pre-standardised
        y_train,
        scale_pos_weight=scale_pos_weight,
    )
    deeptriage.save(str(output_dir / "deeptriage_cn.pkl"))

    # ------------------------------------------------------------------
    # 3. TabNet  (receives raw arrays; BN handles normalisation internally)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training TabNet …")
    tabnet = TabNetWrapper(
        n_d=tabnet_cfg["n_d"],
        n_a=tabnet_cfg["n_a"],
        n_steps=tabnet_cfg["n_steps"],
        gamma=tabnet_cfg["gamma"],
        lambda_sparse=tabnet_cfg["lambda_sparse"],
        seed=random_state,
    )
    tabnet.fit(
        X_train_raw,          # ← raw; TabNet applies its own BN
        y_train,
        eval_set=[(X_val_raw, y_val)],
        max_epochs=tabnet_cfg["max_epochs"],
        batch_size=tabnet_cfg["batch_size"],
        virtual_batch_size=tabnet_cfg["virtual_batch_size"],
    )
    tabnet.save(str(output_dir / "tabnet"))

    # ------------------------------------------------------------------
    # 4. Vitals-only XGBoost  (receives standardised arrays)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Vitals-only XGBoost …")
    vit_xgb = VitalsOnlyXGBoost(
        n_estimators=xgb_cfg["n_estimators"],
        learning_rate=xgb_cfg["learning_rate"],
        max_depth=xgb_cfg["max_depth"],
        subsample=xgb_cfg["subsample"],
        colsample_bytree=xgb_cfg["colsample_bytree"],
        random_state=random_state,
    )
    vit_xgb.fit(X_train_std, y_train)   # ← standardised
    joblib.dump(vit_xgb, output_dir / "vitals_only_xgb.pkl")

    # ------------------------------------------------------------------
    # 5. Random Forest  (receives standardised arrays)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Random Forest …")
    rf = RandomForestBaseline(
        n_estimators=200, max_depth=6, random_state=random_state
    )
    rf.fit(X_train_std, y_train)        # ← standardised
    joblib.dump(rf, output_dir / "random_forest.pkl")

    # ------------------------------------------------------------------
    # 6. Text-only Logistic Regression  (BERT embeddings, standardised)
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Training Text-only Logistic Regression …")
    # Re-use the already-loaded text_encoder
    train_emb = text_encoder.encode(train_texts.tolist())
    text_lr = TextOnlyLogisticRegression(random_state=random_state)
    text_lr.fit(train_emb, y_train)
    joblib.dump(text_lr, output_dir / "text_only_lr.pkl")

    # Persist the text encoder separately so evaluate / figure scripts
    # can reload it without running HuggingFace download again.
    joblib.dump(text_encoder, output_dir / "text_encoder.pkl")

    # ------------------------------------------------------------------
    # 7. Clinical scores on validation set
    # ------------------------------------------------------------------
    logger.info("=" * 60)
    logger.info("Computing clinical benchmark scores on validation set …")

    def _safe(col: str) -> np.ndarray:
        """Return column as float array (NaN where missing)."""
        return val_df[col].values.astype(float)

    val_scores = pd.DataFrame({
        "NEWS2": compute_news2(
            rr=_safe("respiratory_rate"),
            spo2=_safe("spo2"),
            temp=_safe("temperature"),
            sbp=_safe("sbp"),
            hr=_safe("heart_rate"),
        ),
        "MEWS": compute_mews(
            rr=_safe("respiratory_rate"),
            hr=_safe("heart_rate"),
            sbp=_safe("sbp"),
            temp=_safe("temperature"),
        ),
        "ESI": compute_esi_level(
            age=_safe("age"),
            hr=_safe("heart_rate"),
            rr=_safe("respiratory_rate"),
            sbp=_safe("sbp"),
            spo2=_safe("spo2"),
            temp=_safe("temperature"),
        ),
    })
    val_scores.to_csv(output_dir / "clinical_scores_val.csv", index=False)
    logger.info(f"Clinical scores saved → {output_dir / 'clinical_scores_val.csv'}")

    logger.info("=" * 60)
    logger.info("All models trained and saved successfully.")


if __name__ == "__main__":
    main()
