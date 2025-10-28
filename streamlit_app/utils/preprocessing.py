from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import RobustScaler

from .constants import P_WAVE_FEATURES


@dataclass
class PreprocessingArtifacts:
    """Container for fitted preprocessing steps."""

    scaler: RobustScaler
    imputer: SimpleImputer
    selector: SelectKBest

    def transform(self, features: pd.DataFrame) -> np.ndarray:
        """Apply the preprocessing pipeline (log1p, scale, impute, select)."""

        log_features = np.log1p(features)
        scaled = self.scaler.transform(log_features)
        imputed = self.imputer.transform(scaled)
        selected = self.selector.transform(imputed)
        return selected


def fit_preprocessing_pipeline(
    features: pd.DataFrame, target_log: pd.Series
) -> PreprocessingArtifacts:
    """Fit preprocessing pipeline using the training subset."""

    log_features = np.log1p(features)
    scaler = RobustScaler().fit(log_features)
    scaled = scaler.transform(log_features)

    imputer = SimpleImputer(strategy="mean").fit(scaled)
    imputed = imputer.transform(scaled)

    selector = SelectKBest(score_func=f_regression, k="all").fit(imputed, target_log)

    return PreprocessingArtifacts(scaler=scaler, imputer=imputer, selector=selector)


def build_feature_frame(raw_features: Dict[str, float]) -> pd.DataFrame:
    """Validate incoming feature dictionary and produce a DataFrame."""

    missing: Iterable[str] = [f for f in P_WAVE_FEATURES if f not in raw_features]
    if missing:
        raise ValueError(f"Missing required features: {missing}")

    df = pd.DataFrame([{feature: float(raw_features[feature]) for feature in P_WAVE_FEATURES}])
    return df


def log_transform_target(target: pd.Series) -> Tuple[pd.Series, pd.Series]:
    """Return log1p target and original target."""

    target_log = np.log1p(target)
    return target_log, target


def save_artifacts(artifacts: PreprocessingArtifacts, path: str) -> None:
    joblib.dump(
        {
            "scaler": artifacts.scaler,
            "imputer": artifacts.imputer,
            "selector": artifacts.selector,
        },
        path,
    )


def load_artifacts(path: str) -> PreprocessingArtifacts:
    bundle = joblib.load(path)
    if isinstance(bundle, PreprocessingArtifacts):
        return bundle
    if not isinstance(bundle, dict):
        raise TypeError("Unexpected artifact format. Expected dict or PreprocessingArtifacts instance.")
    return PreprocessingArtifacts(
        scaler=bundle["scaler"],
        imputer=bundle["imputer"],
        selector=bundle["selector"],
    )
