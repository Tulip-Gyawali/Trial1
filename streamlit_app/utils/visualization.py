from __future__ import annotations

import base64
import io
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from obspy import Trace
from plotly.subplots import make_subplots

from .constants import INTENSITY_BINS


def _array_to_base64(fig) -> str:
    buf = io.BytesIO()
    fig.write_image(buf, format="png")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def pga_to_intensity(pga_g: float) -> Tuple[str, str]:
    for lo, hi, label, color in INTENSITY_BINS:
        if lo <= pga_g < hi:
            return label, color
    return "Extreme", "#111111"


def build_feature_table(features: Dict[str, float]) -> pd.DataFrame:
    return (
        pd.DataFrame(list(features.items()), columns=["Feature", "Value"])
        .sort_values("Feature")
        .reset_index(drop=True)
    )


def build_waveform_plot(trace: Trace, p_index: int, window_samples: int) -> go.Figure:
    dt = trace.stats.delta
    times = np.arange(0, trace.stats.npts) * dt
    p_time = p_index * dt

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=times, y=trace.data, mode="lines", name="Seismogram", line=dict(width=1.0))
    )
    fig.add_vline(x=p_time, line_dash="dash", line_color="red", name="P-pick")
    fig.add_shape(
        type="rect",
        x0=p_time,
        x1=p_time + window_samples * dt,
        y0=float(np.min(trace.data)),
        y1=float(np.max(trace.data)),
        fillcolor="rgba(255,0,0,0.1)",
        line=dict(width=0),
    )
    fig.update_layout(
        title="Seismogram with P-window",
        xaxis_title="Time (s)",
        yaxis_title="Amplitude",
        height=300,
        margin=dict(l=40, r=40, t=50, b=40),
    )
    return fig


def build_p_window_plots(p_window: np.ndarray, sampling_rate: float) -> go.Figure:
    dt = 1.0 / sampling_rate
    times = np.arange(0, len(p_window)) * dt

    fig = make_subplots(rows=2, cols=1, shared_xaxes=False, vertical_spacing=0.08)

    fig.add_trace(
        go.Scatter(x=times, y=p_window, mode="lines", name="P-window"),
        row=1,
        col=1,
    )

    freqs = np.fft.rfftfreq(len(p_window), d=dt)
    psd = np.abs(np.fft.rfft(p_window)) ** 2

    fig.add_trace(
        go.Scatter(x=freqs, y=psd, mode="lines", name="FFT Power"),
        row=2,
        col=1,
    )

    fig.update_yaxes(title_text="Amplitude", row=1, col=1)
    fig.update_xaxes(title_text="Time (s)", row=1, col=1)
    fig.update_yaxes(title_text="Power", type="log", row=2, col=1)
    fig.update_xaxes(title_text="Frequency (Hz)", row=2, col=1)

    fig.update_layout(height=450, margin=dict(l=40, r=40, t=50, b=40))
    return fig
