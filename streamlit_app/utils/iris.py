from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
from obspy import Stream, Trace, UTCDateTime
from obspy.clients.fdsn import Client
from obspy.signal.trigger import classic_sta_lta, trigger_onset

from .constants import DEFAULT_IRIS_NETWORK, DEFAULT_IRIS_STATIONS, P_WAVE_FEATURES


@dataclass
class SeismogramSample:
    trace: Trace
    p_window: np.ndarray
    p_index: int
    features: Dict[str, float]
    metadata: Dict[str, object]


class IRISSeismogramFetcher:
    def __init__(
        self,
        network: str = DEFAULT_IRIS_NETWORK,
        stations: Optional[List[str]] = None,
        client: Optional[Client] = None,
    ) -> None:
        self.network = network
        self.stations = stations or DEFAULT_IRIS_STATIONS
        self.client = client or Client("IRIS")

    def fetch_sample(
        self, year_choices: Optional[List[int]] = None, max_attempts: int = 8
    ) -> Optional[SeismogramSample]:
        year_choices = year_choices or [2022, 2023, 2024]

        for _ in range(max_attempts):
            try:
                sample = self._try_fetch(year_choices)
                if sample:
                    return sample
            except Exception:
                continue
        return None

    def _try_fetch(self, year_choices: List[int]) -> Optional[SeismogramSample]:
        year = random.choice(year_choices)
        start_time = UTCDateTime(
            dt.datetime(
                year,
                random.randint(1, 12),
                random.randint(1, 25),
                random.randint(0, 21),
                0,
                0,
            )
        )
        end_time = start_time + 2 * 3600

        trace = self._find_trace(start_time, end_time)
        if trace is None:
            return None

        trace = trace.copy()
        trace.detrend("demean")
        trace.filter("bandpass", freqmin=0.5, freqmax=20.0)

        dt_seconds = trace.stats.delta
        cft = classic_sta_lta(trace.data, int(1 / dt_seconds), int(10 / dt_seconds))
        picks = trigger_onset(cft, 2.5, 1.0)
        if len(picks) == 0:
            return None

        p_index = int(picks[0][0])
        window_samples = int(2.0 / dt_seconds)
        p_window = trace.data[p_index : p_index + window_samples]
        if len(p_window) < 10:
            return None

        features = self._compute_pwave_features(p_window, dt_seconds)

        metadata = {
            "station": trace.stats.station,
            "network": trace.stats.network,
            "starttime": str(start_time),
            "sampling_rate": trace.stats.sampling_rate,
            "p_index": p_index,
            "window_samples": len(p_window),
        }

        # Attempt to enrich with station coordinates from IRIS metadata.
        try:
            inv = self.client.get_stations(
                network=metadata["network"],
                station=metadata["station"],
                level="station",
                starttime=start_time,
                endtime=end_time,
            )
            if inv and inv[0].stations:
                station_meta = inv[0].stations[0]
                metadata["latitude"] = getattr(station_meta, "latitude", None)
                metadata["longitude"] = getattr(station_meta, "longitude", None)
                metadata["elevation_m"] = getattr(station_meta, "elevation", None)
        except Exception:
            metadata.setdefault("latitude", None)
            metadata.setdefault("longitude", None)

        return SeismogramSample(
            trace=trace,
            p_window=p_window,
            p_index=p_index,
            features=features,
            metadata=metadata,
        )

    def _find_trace(self, start_time: UTCDateTime, end_time: UTCDateTime) -> Optional[Trace]:
        for station in self.stations:
            try:
                stream: Stream = self.client.get_waveforms(
                    self.network, station, "*", "BHZ", start_time, end_time
                )
                if stream and len(stream) > 0:
                    return stream[0]
            except Exception:
                continue
        return None

    def _compute_pwave_features(self, window: np.ndarray, dt_seconds: float) -> Dict[str, float]:
        if len(window) == 0:
            return {feature: float("nan") for feature in P_WAVE_FEATURES}

        duration = len(window) * dt_seconds
        gradient = np.gradient(window) / dt_seconds
        gradient_abs = np.abs(gradient)

        def ddt(signal: np.ndarray) -> float:
            return float(np.mean(np.abs(np.gradient(signal)))) if len(signal) > 1 else 0.0

        PDd = float(np.max(window) - np.min(window))
        PVd = float(np.max(gradient_abs) if len(gradient_abs) > 0 else 0.0)
        PAd = float(np.mean(np.abs(window)))
        PDt = float(np.max(window))
        PVt = float(np.max(gradient)) if len(gradient) > 0 else 0.0
        PAt = float(np.sqrt(np.mean(window**2)))

        tauPd = float(duration / PDd) if PDd != 0 else 0.0
        tauPt = float(duration / PDt) if PDt != 0 else 0.0

        features: Dict[str, float] = {
            "pkev12": float(np.sum(window**2) / len(window)),
            "pkev23": float(np.sum(np.abs(window)) / len(window)),
            "durP": duration,
            "tauPd": tauPd,
            "tauPt": tauPt,
            "PDd": PDd,
            "PVd": PVd,
            "PAd": PAd,
            "PDt": PDt,
            "PVt": PVt,
            "PAt": PAt,
            "ddt_PDd": ddt(window),
            "ddt_PVd": ddt(gradient),
            "ddt_PAd": ddt(np.abs(window)),
            "ddt_PDt": ddt(np.maximum(window, 0)),
            "ddt_PVt": ddt(gradient),
            "ddt_PAt": ddt(window**2),
        }

        return features
