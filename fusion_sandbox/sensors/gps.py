from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .faults import in_windows, outlier_offset
from .timing import available_time, sample_times


class GPSSensor:
    name = "gps"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", True):
            return []
        rate = float(self.config.get("rate_hz", 1.0))
        pos_noise = float(self.config.get("position_noise_std_m", 1.8))
        vel_noise = float(self.config.get("velocity_noise_std_mps", 0.25))
        latency = float(self.config.get("latency_s", 0.0))
        dropouts = self.config.get("dropout_windows_s", [])
        bursts = self.config.get("outlier_bursts", [])
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            dropout = in_windows(t, dropouts)
            pos = np.array([row["px"], row["py"], row["pz"]], dtype=float)
            vel = np.array([row["vx"], row["vy"], row["vz"]], dtype=float)
            offset, is_outlier = outlier_offset(t, bursts, "position_offset_m", 3)
            measured_pos = pos + rng.normal(0.0, pos_noise, 3) + offset
            measured_vel = vel + rng.normal(0.0, vel_noise, 3)
            events.append(
                {
                    "time_s": available_time(t, latency),
                    "measurement_time_s": t,
                    "sensor": "gps",
                    "kind": "position_velocity",
                    "valid": 0.0 if dropout else 1.0,
                    "is_dropout": 1.0 if dropout else 0.0,
                    "is_outlier": 1.0 if is_outlier else 0.0,
                    "x": float(measured_pos[0]),
                    "y": float(measured_pos[1]),
                    "z": float(measured_pos[2]),
                    "vx": float(measured_vel[0]),
                    "vy": float(measured_vel[1]),
                    "vz": float(measured_vel[2]),
                    "latency_s": latency,
                }
            )
        return events
