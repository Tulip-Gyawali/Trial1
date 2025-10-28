# Streamlit PGA Prediction Project Guide

## Overview

- **Tech stack**: Streamlit + Python 3.11, XGBoost, scikit-learn, Plotly, Folium, ObsPy
- **Entry points**:
  - `streamlit_app/app.py` – launches the interactive dashboard
  - `streamlit_app/train.py` – trains the Optuna-tuned XGBoost pipeline (identical preprocessing to notebook)
- **Model artifacts**: Persisted under `streamlit_app/models/` (`xgb_eew_final.joblib`, `preproc_objects.joblib`, `metrics_summary.parquet`). The Streamlit app will retrain automatically if artifacts are missing and a dataset is available.

## Commands

```bash
# Install dependencies (use your preferred environment manager)
pip install -r requirements.txt

# Train pipeline with original dataset (outputs saved in streamlit_app/models/)
python -m streamlit_app.train --data data/EEW_features_2024-10-21.csv

# Launch Streamlit application
streamlit run streamlit_app/app.py
```

> **Note**: The execution sandbox in Youware currently lacks Python runtime tooling; commands above are for local development or remote environments with Python installed.

## Architecture Overview

```
streamlit_app/
├── app.py                # Streamlit UI, seismogram fetch, prediction, visualization
├── train.py              # CLI to retrain model using dataset loader/preprocessing
├── models/               # Saved joblib artifacts + optional metrics parquet
├── utils/
│   ├── constants.py      # P-wave feature list, XGB hyperparameters, intensity bins
│   ├── datasets.py       # Dataset loading/cleaning (matches notebook logic)
│   ├── iris.py           # IRIS seismogram fetcher + feature extraction
│   ├── metrics.py        # Helpers to display evaluation metrics
│   ├── model.py          # Pipeline wrapper, save/load helpers
│   ├── preprocessing.py  # Preprocessing pipeline (log1p, RobustScaler, etc.)
│   └── visualization.py  # Plotly/Folium visualization helpers
└── __init__.py
```

### Pipeline details

- Preserves notebook behavior: log1p transformation on features and target, RobustScaler, SimpleImputer (`mean`), SelectKBest (`f_regression`, all features), and Optuna-tuned XGBRegressor.
- Train/test splitting uses `StratifiedShuffleSplit` on log-transformed target deciles, exactly mirroring the notebook.
- Predictions returned in both cm/s² and g units; intensity categories map to visual badges and Folium circle markers.

### IRIS integration

- `IRISSeismogramFetcher` queries the IRIS FDSN service (default network IU, stations ANMO/COR/MAJO/KBL).
- STA/LTA detection replicates the notebook thresholds (classic STA/LTA with 1s/10s windows, 2.5/1.0 triggers).
- P-wave window (2 seconds) feeds feature computation; station metadata (lat/lon/elevation) is requested to support map displays.
- Network availability may fluctuate; retries are built in (up to 8 attempts).

## Dataset

- `data/EEW_features_2024-10-21.csv` is the canonical dataset.
- Loader trims stray header rows, strips column names, coerces numerics, median-imputes numeric NaNs, and filters rows where every P-wave feature is strictly positive.

## Metrics persistence

- Training CLI writes `metrics_summary.parquet` capturing log/raw metrics across train/val/test splits.
- Streamlit app reads the parquet (if present) and caches metrics in session state for display.

## Additional notes

- Streamlit caching (`st.cache_data`, `st.cache_resource`) ensures dataset loads and model instantiation are efficient across reruns.
- Folium map rendering requires coordinates; if IRIS metadata omits them, the app falls back to textual summaries.
- Retry seismogram fetch if IRIS responds slowly or returns no picks.
