from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import joblib
import numpy as np
import xgboost as xgb

from .constants import BEST_XGB_PARAMS, P_WAVE_FEATURES
from .preprocessing import PreprocessingArtifacts, build_feature_frame


@dataclass
class XGBPipeline:
    """Wraps preprocessing artifacts and the trained XGB model."""

    artifacts: PreprocessingArtifacts
    model: xgb.XGBRegressor

    def predict_pga(self, features: Dict[str, float]) -> Dict[str, float]:
        """Return prediction in cm/s^2 and g units."""

        frame = build_feature_frame(features)
        transformed = self.artifacts.transform(frame)
        y_log = self.model.predict(transformed)
        y_cm = np.expm1(y_log)[0]
        y_g = y_cm / 980.0
        return {"pga_cm_s2": float(y_cm), "pga_g": float(y_g)}


def instantiate_model() -> xgb.XGBRegressor:
    return xgb.XGBRegressor(
        objective="reg:squarederror",
        n_estimators=int(BEST_XGB_PARAMS["n_estimators"]),
        learning_rate=float(BEST_XGB_PARAMS["learning_rate"]),
        max_depth=int(BEST_XGB_PARAMS["max_depth"]),
        subsample=float(BEST_XGB_PARAMS["subsample"]),
        colsample_bytree=float(BEST_XGB_PARAMS["colsample_bytree"]),
        random_state=42,
        verbosity=0,
        tree_method="auto",
    )


def load_pipeline(model_path: Path, artifacts_path: Path) -> XGBPipeline:
    model = joblib.load(model_path)
    artifacts = load_artifacts(artifacts_path)
    if not isinstance(model, xgb.XGBRegressor):
        raise TypeError("Loaded model is not an XGBRegressor")
    return XGBPipeline(artifacts=artifacts, model=model)


def save_pipeline(pipeline: XGBPipeline, model_path: Path, artifacts_path: Path) -> None:
    joblib.dump(pipeline.model, model_path)
    joblib.dump(pipeline.artifacts, artifacts_path)
