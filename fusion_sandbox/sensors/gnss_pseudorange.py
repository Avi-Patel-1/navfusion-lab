from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .faults import in_windows
from .timing import sample_times


class GNSSPseudorangeSensor:
    """Raw pseudorange receiver with deterministic satellite geometry."""

    name = "gnss_pseudorange"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _outlier_offset(self, t: float) -> tuple[float, bool]:
        for burst in self.config.get("outlier_bursts", []):
            start = float(burst.get("start_s", 0.0))
            duration = float(burst.get("duration_s", 0.0))
            if start <= t <= start + duration:
                return float(burst.get("pseudorange_offset_m", 0.0)), True
        return 0.0, False

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 1.0))
        noise = float(self.config.get("pseudorange_noise_std_m", 1.2))
        satellites = [np.asarray(sat, dtype=float).reshape(3) for sat in self.config.get("satellites_m", [])]
        clock_bias = float(self.config.get("receiver_clock_bias_m", 0.0))
        clock_correction = float(self.config.get("clock_bias_correction_m", 0.0))
        clock_drift = float(self.config.get("clock_drift_mps", 0.0))
        packet_loss = float(self.config.get("packet_loss_probability", 0.0))
        jitter = float(self.config.get("timestamp_jitter_std_s", 0.0))
        windows = self.config.get("dropout_windows_s", [])
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            pos = np.array([row["px"], row["py"], row["pz"]], dtype=float)
            current_clock_bias = clock_bias + clock_drift * t
            lost_epoch = bool(packet_loss > 0.0 and rng.uniform() < packet_loss)
            in_dropout = lost_epoch or in_windows(t, windows)
            availability = max(0.0, t + (float(rng.normal(0.0, jitter)) if jitter > 0.0 else 0.0))
            outlier_offset, is_outlier = self._outlier_offset(t)
            for index, satellite in enumerate(satellites):
                true_range = float(np.linalg.norm(pos - satellite))
                pseudorange = true_range + current_clock_bias + outlier_offset + float(rng.normal(0.0, noise))
                events.append(
                    {
                        "time_s": round(availability, 10),
                        "measurement_time_s": t,
                        "sensor": "gnss_pseudorange",
                        "kind": "pseudorange",
                        "valid": 0.0 if in_dropout else 1.0,
                        "is_dropout": 1.0 if in_dropout else 0.0,
                        "is_outlier": 1.0 if is_outlier else 0.0,
                        "satellite_id": float(index),
                        "satellite_x": float(satellite[0]),
                        "satellite_y": float(satellite[1]),
                        "satellite_z": float(satellite[2]),
                        "true_range_m": true_range,
                        "pseudorange_m": pseudorange,
                        "clock_bias_true_m": current_clock_bias,
                        "clock_bias_correction_m": clock_correction,
                    }
                )
        return events
