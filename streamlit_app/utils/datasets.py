from __future__ import annotations

from pathlib import Path
from typing import Tuple

import pandas as pd

from .constants import P_WAVE_FEATURES


def load_eew_dataset(csv_path: Path) -> pd.DataFrame:
    """Load and clean the EEW feature dataset."""

    if not csv_path.exists():
        raise FileNotFoundError(f"Dataset not found at {csv_path}")

    try:
        df = pd.read_csv(csv_path, skiprows=[1])
    except Exception:
        df = pd.read_csv(csv_path)

    df.columns = df.columns.str.strip()

    # Force numeric conversion for all relevant columns and fill missing values.
    for column in df.columns:
        if column not in {"filename", "date", "time"}:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    # Replace NaNs in numeric columns with column medians.
    numeric_cols = df.select_dtypes(include=["number"]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())

    # Keep only rows where every P-wave feature is strictly positive.
    df = df[(df[P_WAVE_FEATURES] > 0).all(axis=1)]
    df = df.drop_duplicates(subset=P_WAVE_FEATURES + ["PGA"])

    return df.reset_index(drop=True)


def get_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """Return feature matrix X and target vector y."""

    missing_columns = set(P_WAVE_FEATURES) - set(df.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing required features: {missing_columns}")

    if "PGA" not in df.columns:
        raise ValueError("Dataset must contain a 'PGA' column for the target variable")

    X = df[P_WAVE_FEATURES].copy()
    y = df["PGA"].astype(float)
    return X, y
