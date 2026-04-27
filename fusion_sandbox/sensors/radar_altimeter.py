from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .faults import in_windows
from .timing import sample_times


class RadarAltimeterSensor:
    """Terrain-relative altitude source with simple warmup, packet loss, and outliers."""

    name = "radar_altimeter"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _outlier_offset(self, t: float) -> tuple[float, bool]:
        for burst in self.config.get("outlier_bursts", []):
            start = float(burst.get("start_s", 0.0))
            duration = float(burst.get("duration_s", 0.0))
            if start <= t <= start + duration:
                return float(burst.get("altitude_offset_m", 0.0)), True
        return 0.0, False

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 8.0))
        noise = float(self.config.get("altitude_noise_std_m", 0.35))
        terrain = float(self.config.get("terrain_altitude_m", 0.0))
        bias = float(self.config.get("bias_m", 0.0))
        warmup = float(self.config.get("warmup_s", 0.0))
        packet_loss = float(self.config.get("packet_loss_probability", 0.0))
        jitter = float(self.config.get("timestamp_jitter_std_s", 0.0))
        windows = self.config.get("dropout_windows_s", [])
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            warm = t < warmup
            lost = bool(packet_loss > 0.0 and rng.uniform() < packet_loss)
            in_dropout = warm or lost or in_windows(t, windows)
            offset, is_outlier = self._outlier_offset(t)
            availability = max(0.0, t + (float(rng.normal(0.0, jitter)) if jitter > 0.0 else 0.0))
            true_agl = float(row["pz"]) - terrain
            measured = true_agl + bias + offset + float(rng.normal(0.0, noise))
            events.append(
                {
                    "time_s": round(availability, 10),
                    "measurement_time_s": t,
                    "sensor": "radar_altimeter",
                    "kind": "altitude_agl",
                    "valid": 0.0 if in_dropout else 1.0,
                    "is_dropout": 1.0 if in_dropout else 0.0,
                    "is_outlier": 1.0 if is_outlier else 0.0,
                    "altitude_agl_m": measured,
                    "terrain_altitude_m": terrain,
                    "true_altitude_agl_m": true_agl,
                    "bias_true_m": bias,
                }
            )
        return events
