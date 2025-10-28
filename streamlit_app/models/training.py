from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import StratifiedShuffleSplit

from ..utils.constants import P_WAVE_FEATURES
from ..utils.datasets import get_features_and_target
from ..utils.model import instantiate_model
from ..utils.preprocessing import (
    PreprocessingArtifacts,
    fit_preprocessing_pipeline,
    log_transform_target,
)


@dataclass
class TrainingOutputs:
    model: xgb.XGBRegressor
    artifacts: PreprocessingArtifacts
    metrics_log: dict
    metrics_raw: dict


@dataclass
class TrainValTestSplit:
    X_train: pd.DataFrame
    y_train: pd.Series
    X_val: pd.DataFrame
    y_val: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series


def stratified_splits(X: pd.DataFrame, y_log: pd.Series) -> TrainValTestSplit:
    bins = pd.qcut(y_log, q=10, labels=False, duplicates="drop")
    sss1 = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    train_idx, temp_idx = next(sss1.split(X, bins))

    sss2 = StratifiedShuffleSplit(n_splits=1, train_size=0.5, random_state=42)
    val_idx, test_idx = next(sss2.split(X.iloc[temp_idx], bins.iloc[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    return TrainValTestSplit(
        X_train=X.iloc[train_idx],
        y_train=y_log.iloc[train_idx],
        X_val=X.iloc[val_idx],
        y_val=y_log.iloc[val_idx],
        X_test=X.iloc[test_idx],
        y_test=y_log.iloc[test_idx],
    )


def evaluate_predictions(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> dict:
    return {
        "R2": r2_score(y_true_log, y_pred_log),
        "MAE": mean_absolute_error(y_true_log, y_pred_log),
        "RMSE": np.sqrt(mean_squared_error(y_true_log, y_pred_log)),
    }


def evaluate_pipeline(
    model: xgb.XGBRegressor,
    artifacts: PreprocessingArtifacts,
    X: pd.DataFrame,
    y_log: pd.Series,
    y_raw: pd.Series,
) -> tuple[dict, dict]:
    transformed = artifacts.transform(X)
    y_pred_log = model.predict(transformed)
    y_pred_raw = np.expm1(y_pred_log)

    metrics_log = evaluate_predictions(y_log, y_pred_log)
    metrics_raw = {
        "R2": r2_score(y_raw, y_pred_raw),
        "MAE": mean_absolute_error(y_raw, y_pred_raw),
        "RMSE": np.sqrt(mean_squared_error(y_raw, y_pred_raw)),
    }

    return metrics_log, metrics_raw


def train_model(df: pd.DataFrame, save_dir: Optional[Path] = None) -> TrainingOutputs:
    X, y_raw = get_features_and_target(df)
    y_log, _ = log_transform_target(y_raw)

    splits = stratified_splits(X, y_log)
    artifacts = fit_preprocessing_pipeline(splits.X_train, splits.y_train)

    transformed_train = artifacts.transform(splits.X_train)
    model = instantiate_model()
    model.fit(transformed_train, splits.y_train)

    # Evaluate
    metrics = {}
    metrics_raw = {}
    for label, X_split, y_split_log in [
        ("train", splits.X_train, splits.y_train),
        ("val", splits.X_val, splits.y_val),
        ("test", splits.X_test, splits.y_test),
    ]:
        y_split_raw = np.expm1(y_split_log)
        log_metrics, raw_metrics = evaluate_pipeline(model, artifacts, X_split, y_split_log, y_split_raw)
        metrics[label] = log_metrics
        metrics_raw[label] = raw_metrics

    outputs = TrainingOutputs(
        model=model,
        artifacts=artifacts,
        metrics_log=metrics,
        metrics_raw=metrics_raw,
    )

    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, save_dir / "xgb_eew_final.joblib")
        joblib.dump(artifacts, save_dir / "preproc_objects.joblib")

    return outputs
