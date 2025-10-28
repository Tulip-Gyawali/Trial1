from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from streamlit.components.v1 import html

from .models.training import train_model
from .utils.constants import INTENSITY_BINS, P_WAVE_FEATURES
from .utils.datasets import get_features_and_target, load_eew_dataset
from .utils.iris import IRISSeismogramFetcher, SeismogramSample
from .utils.metrics import build_metrics_table
from .utils.model import XGBPipeline, load_pipeline
from .utils.visualization import (
    build_feature_table,
    build_p_window_plots,
    build_waveform_plot,
    pga_to_intensity,
)

BASE_DIR = Path(__file__).resolve().parent
DEFAULT_DATASET = Path("data/EEW_features_2024-10-21.csv")
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "xgb_eew_final.joblib"
ARTIFACTS_PATH = MODEL_DIR / "preproc_objects.joblib"


st.set_page_config(
    page_title="PGA Prediction from P-wave Features",
    layout="wide",
    page_icon="🌐",
)


@st.cache_data(show_spinner=False)
def load_dataset(path: Path) -> pd.DataFrame:
    return load_eew_dataset(path)


@st.cache_resource(show_spinner=False)
def load_or_train_pipeline(dataset_path: Path) -> Optional[XGBPipeline]:
    if MODEL_PATH.exists() and ARTIFACTS_PATH.exists():
        pipeline = load_pipeline(MODEL_PATH, ARTIFACTS_PATH)
        metrics_path = MODEL_DIR / "metrics_summary.parquet"
        if metrics_path.exists():
            metrics_df = pd.read_parquet(metrics_path)
            st.session_state["training_metrics_log"] = {
                row.split: {
                    "R2": row.R2,
                    "MAE": row.MAE,
                    "RMSE": row.RMSE,
                }
                for row in metrics_df[metrics_df.scale == "log"].itertuples()
            }
            st.session_state["training_metrics_raw"] = {
                row.split: {
                    "R2": row.R2,
                    "MAE": row.MAE,
                    "RMSE": row.RMSE,
                }
                for row in metrics_df[metrics_df.scale == "raw"].itertuples()
            }
        return pipeline

    if not dataset_path.exists():
        return None

    df = load_eew_dataset(dataset_path)
    outputs = train_model(df, save_dir=MODEL_DIR)
    st.session_state["training_metrics_log"] = outputs.metrics_log
    st.session_state["training_metrics_raw"] = outputs.metrics_raw
    return XGBPipeline(artifacts=outputs.artifacts, model=outputs.model)


@st.cache_data(show_spinner=False)
def cached_metrics() -> Dict[str, Dict[str, Dict[str, float]]]:
    metrics = {
        "log": st.session_state.get("training_metrics_log", {}),
        "raw": st.session_state.get("training_metrics_raw", {}),
    }
    return metrics


def ensure_pipeline(dataset_path: Path) -> Optional[XGBPipeline]:
    try:
        pipeline = load_or_train_pipeline(dataset_path)
        return pipeline
    except Exception as exc:
        st.error(f"Failed to load or train pipeline: {exc}")
        return None


def render_intensity_badge(pga_g: float) -> str:
    label, color = pga_to_intensity(pga_g)
    return f"""
    <div style="display:flex;align-items:center;gap:10px;margin-top:6px;">
      <span style="width:14px;height:14px;border-radius:4px;background:{color};border:1px solid #333;"></span>
      <div>
        <div style="font-weight:600;font-size:15px;color:#111;">{label}</div>
        <div style="font-size:13px;color:#555;">PGA: {pga_g:.5g} g</div>
      </div>
    </div>
    """


def embed_folium_map(sample: SeismogramSample, prediction: Dict[str, float]) -> None:
    lat = sample.metadata.get("latitude")
    lon = sample.metadata.get("longitude")

    if lat is None or lon is None:
        st.info(
            "Station coordinates unavailable from IRIS metadata. Map rendering skipped.",
            icon="ℹ️",
        )
        return

    label, color = pga_to_intensity(prediction["pga_g"])
    popup_html = f"""
        <h4>Station {sample.metadata.get('station')}</h4>
        <p>Network: {sample.metadata.get('network')}</p>
        <p>PGA: {prediction['pga_cm_s2']:.2f} cm/s² ({prediction['pga_g']:.5f} g)</p>
        <p>Intensity: <strong style='color:{color}'>{label}</strong></p>
    """

    fmap = folium.Map(location=[lat, lon], zoom_start=6, tiles="CartoDB positron")
    folium.CircleMarker(
        location=[lat, lon],
        radius=10,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.85,
        popup=popup_html,
    ).add_to(fmap)

    folium.Circle(
        location=[lat, lon],
        radius=max(5_000, min(80_000, prediction["pga_g"] * 200_000)),
        color=color,
        fill=True,
        fill_opacity=0.1,
    ).add_to(fmap)

    legend_items = []
    for lo, hi, name, swatch in INTENSITY_BINS:
        rng = f"{lo:.3f} – {hi:.3f}" if hi < 10 else f"> {lo:.3f}"
        legend_items.append(
            f"<div style='display:flex;align-items:center;gap:6px;margin-top:4px;'>"
            f"<span style='display:inline-block;width:16px;height:10px;background:{swatch};border:1px solid #222;'></span>"
            f"<span style='font-size:12px;color:#222'>{name} ({rng} g)</span>"
            "</div>"
        )
    legend_html = (
        "<div style=\"position:fixed;bottom:30px;left:10px;padding:10px;background:white;"
        "border:1px solid #444;border-radius:6px;font-size:12px;z-index:9999;\">"
        "<strong>Intensity classes</strong>"
        + "".join(legend_items)
        + "</div>"
    )
    fmap.get_root().html.add_child(folium.Element(legend_html))

    map_html = fmap.get_root().render()
    html(map_html, height=420)


# Sidebar controls
st.sidebar.header("Configuration")
dataset_path = st.sidebar.text_input(
    "Dataset path",
    value=str(DEFAULT_DATASET),
    help="CSV used for model training and exploratory analysis.",
)
dataset_file = Path(dataset_path)

train_option = st.sidebar.selectbox(
    "When artifacts are missing",
    (
        "Train automatically",
        "Require manual training",
    ),
)

if st.sidebar.button("Force retrain model"):
    if dataset_file.exists():
        st.cache_resource.clear()
        st.cache_data.clear()
        pipeline = load_or_train_pipeline(dataset_file)
        if pipeline:
            st.success("Model retrained using the provided dataset.")
    else:
        st.error("Dataset path is invalid; cannot retrain.")

# Load dataset & pipeline
with st.spinner("Loading dataset..."):
    try:
        dataset = load_dataset(dataset_file)
    except Exception as exc:
        dataset = None
        st.error(f"Unable to load dataset: {exc}")

pipeline = None
if MODEL_PATH.exists() and ARTIFACTS_PATH.exists():
    pipeline = load_or_train_pipeline(dataset_file)
elif train_option == "Train automatically" and dataset is not None:
    pipeline = load_or_train_pipeline(dataset_file)
else:
    st.warning(
        "Model artifacts not found. Provide them under streamlit_app/models or enable automatic training.",
        icon="⚠️",
    )

st.title("PGA Prediction from P-wave Features")
st.caption(
    "Streamlit interface replicating the Optuna-tuned XGBoost pipeline for PGA estimation."
)

col_a, col_b = st.columns([2, 1])
with col_a:
    st.subheader("Dataset preview")
    if dataset is not None:
        st.dataframe(dataset.head(25), use_container_width=True)
    else:
        st.info("Load a dataset to unlock training and exploratory sections.")

with col_b:
    st.subheader("Feature metadata")
    st.write(
        "This model expects the following 17 P-wave features computed from the seismogram P-window:"
    )
    st.json(P_WAVE_FEATURES)
    st.write("Model hyperparameters are fixed to the Optuna-tuned XGBoost configuration.")

# Metrics section
metrics_cache = cached_metrics()
if metrics_cache.get("log"):
    st.subheader("Evaluation metrics (log scale)")
    st.dataframe(build_metrics_table(metrics_cache["log"]).style.format({"R2": "{:.4f}", "MAE": "{:.4f}", "RMSE": "{:.4f}"}))

if metrics_cache.get("raw"):
    st.subheader("Evaluation metrics (raw scale)")
    st.dataframe(build_metrics_table(metrics_cache["raw"]).style.format({"R2": "{:.4f}", "MAE": "{:.2f}", "RMSE": "{:.2f}"}))

# Manual feature input or sample selection
st.header("Predict PGA from feature set")
col1, col2 = st.columns([1, 1])

with col1:
    chosen_row_idx = None
    if dataset is not None:
        st.write("Select a row from the dataset to prefill features:")
        chosen_row_idx = st.number_input(
            "Row index",
            min_value=0,
            max_value=len(dataset) - 1,
            value=0,
        )

    manual_features: Dict[str, float] = {}
    default_values = dataset.loc[chosen_row_idx, P_WAVE_FEATURES].to_dict() if dataset is not None else {}

    with st.form("feature_form"):
        for feature in P_WAVE_FEATURES:
            default_val = float(default_values.get(feature, 0.0))
            manual_features[feature] = st.number_input(
                feature,
                value=float(default_val),
                format="%.6f",
            )
        submitted = st.form_submit_button("Predict from features")

    if submitted:
        if pipeline is None:
            st.error("Model pipeline not available. Train or load artifacts first.")
        else:
            prediction = pipeline.predict_pga(manual_features)
            st.success(
                f"Predicted PGA: {prediction['pga_cm_s2']:.2f} cm/s² ({prediction['pga_g']:.5f} g)",
                icon="✅",
            )
            st.components.v1.html(render_intensity_badge(prediction["pga_g"]), height=60)

with col2:
    st.write("Download feature payload as JSON:")
    if manual_features:
        st.download_button(
            label="Download feature JSON",
            file_name="p_wave_features.json",
            mime="application/json",
            data=json.dumps(manual_features, indent=2),
        )


# IRIS seismogram section
st.header("Fetch seismogram from IRIS and infer PGA")
fetcher = IRISSeismogramFetcher()
if st.button("Fetch sample seismogram"):
    with st.spinner("Querying IRIS and computing features..."):
        sample = fetcher.fetch_sample()
    if sample is None:
        st.error("Failed to fetch a seismogram. Try again; IRIS availability can fluctuate.")
    else:
        st.session_state["latest_sample"] = sample
        st.experimental_rerun()

sample: Optional[SeismogramSample] = st.session_state.get("latest_sample")
if sample:
    st.subheader("Latest fetched sample")
    meta_cols = st.columns(3)
    meta_cols[0].metric("Station", sample.metadata.get("station"))
    meta_cols[1].metric("Network", sample.metadata.get("network"))
    meta_cols[2].metric("Sampling rate", f"{sample.metadata.get('sampling_rate')} Hz")

    left, right = st.columns([2, 1])
    with left:
        st.plotly_chart(build_waveform_plot(sample.trace, sample.p_index, sample.metadata.get("window_samples", 0)), use_container_width=True)
        st.plotly_chart(build_p_window_plots(sample.p_window, sample.metadata.get("sampling_rate", 1.0)), use_container_width=True)
    with right:
        st.write("Extracted features")
        st.dataframe(build_feature_table(sample.features), use_container_width=True)

    if pipeline is None:
        st.warning("Model pipeline unavailable. Train or load artifacts to evaluate this sample.")
    else:
        prediction = pipeline.predict_pga(sample.features)
        st.success(
            f"Predicted PGA (raw): {prediction['pga_cm_s2']:.2f} cm/s² ({prediction['pga_g']:.5f} g)",
            icon="✅",
        )
        st.components.v1.html(render_intensity_badge(prediction["pga_g"]), height=60)
        embed_folium_map(sample, prediction)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Built with Streamlit, leveraging the original Optuna-tuned XGBoost regression pipeline."
)
