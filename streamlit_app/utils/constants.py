from __future__ import annotations

from typing import Dict, List

# List of P-wave features used by the XGBoost model.
P_WAVE_FEATURES: List[str] = [
    "pkev12",
    "pkev23",
    "durP",
    "tauPd",
    "tauPt",
    "PDd",
    "PVd",
    "PAd",
    "PDt",
    "PVt",
    "PAt",
    "ddt_PDd",
    "ddt_PVd",
    "ddt_PAd",
    "ddt_PDt",
    "ddt_PVt",
    "ddt_PAt",
]

# Hyperparameters discovered via Optuna search in the original notebook.
BEST_XGB_PARAMS: Dict[str, float] = {
    "n_estimators": 776,
    "learning_rate": 0.010590433420511285,
    "max_depth": 6,
    "subsample": 0.666852461341688,
    "colsample_bytree": 0.8724127328229327,
}

# Default IRIS network & stations leveraged in the prototype notebook.
DEFAULT_IRIS_NETWORK = "IU"
DEFAULT_IRIS_STATIONS: List[str] = ["ANMO", "COR", "MAJO", "KBL"]

# Intensity bins (PGA in g) for qualitative interpretation.
INTENSITY_BINS = [
    (0.000, 0.005, "Very Weak", "#2ca02c"),
    (0.005, 0.020, "Weak", "#ffdd57"),
    (0.020, 0.050, "Moderate", "#ff7f0e"),
    (0.050, 0.100, "Strong", "#d62728"),
    (0.100, 10.000, "Very Strong", "#7f3b8b"),
]
