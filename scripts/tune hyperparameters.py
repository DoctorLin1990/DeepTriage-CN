#!/usr/bin/env python3
"""
tune_hyperparameters.py

Hyperparameter tuning script for TabNet and XGBoost models, using grid search
with cross-validation as described in Section 3.6.

Usage:
    python scripts/tune_hyperparameters.py --config config.yaml --model tabnet

This script is computationally intensive; it is not required for quick reproduction
since optimal parameters are already set in config.yaml.
"""

import argparse
import yaml
import sys
import numpy as np
from pathlib import Path
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import roc_auc_score
import logging

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.preprocess import load_and_split_data, preprocess_structured, preprocess_text, get_labels
from src.tabnet_model import TabNetWrapper
import xgboost as xgb

logging.basicConfig(level=logging.INFO)


def tune_tabnet(X_train, y_train, X_val, y_val, random_state):
    param_grid = {
        "n_d": [32, 64],
        "n_a": [32, 64],
        "n_steps": [3, 5, 7],
        "gamma": [1.0, 1.5, 2.0],
        "lambda_sparse": [0.0001, 0.00001],
    }
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        model = TabNetWrapper(**params, seed=random_state)
        model.fit(X_train, y_train,
                  eval_set=[(X_val, y_val)],
                  max_epochs=50, patience=10,
                  batch_size=256, virtual_batch_size=128)
        proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        logging.info(f"TabNet params: {params}, AUROC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
    logging.info(f"Best TabNet params: {best_params}, AUROC: {best_score:.4f}")
    return best_params


def tune_xgboost(X_train, y_train, X_val, y_val, random_state):
    param_grid = {
        "n_estimators": [100, 200],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth": [3, 6, 9],
        "subsample": [0.8, 1.0],
        "colsample_bytree": [0.8, 1.0],
    }
    best_score = 0
    best_params = None
    for params in ParameterGrid(param_grid):
        model = xgb.XGBClassifier(
            **params,
            random_state=random_state,
            verbosity=0,
            scale_pos_weight=(np.sum(y_train == 0) / np.sum(y_train == 1)),
        )
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_val)[:, 1]
        score = roc_auc_score(y_val, proba)
        logging.info(f"XGB params: {params}, AUROC: {score:.4f}")
        if score > best_score:
            best_score = score
            best_params = params
    logging.info(f"Best XGB params: {best_params}, AUROC: {best_score:.4f}")
    return best_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--model", choices=["tabnet", "xgboost", "all"], default="all")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)
    random_state = config["random_state"]

    train_df, val_df = load_and_split_data(
        config["data"]["raw_data_path"],
        config["data"]["training_cutoff_date"],
        config["data"]["validation_start_date"],
        random_state,
    )
    X_train_raw, X_val_raw, _ = preprocess_structured(train_df, val_df)
    _, _ = preprocess_text(train_df, val_df)
    y_train, y_val = get_labels(train_df, val_df)

    if args.model in ["tabnet", "all"]:
        tune_tabnet(X_train_raw, y_train, X_val_raw, y_val, random_state)
    if args.model in ["xgboost", "all"]:
        tune_xgboost(X_train_raw, y_train, X_val_raw, y_val, random_state)


if __name__ == "__main__":
    main()