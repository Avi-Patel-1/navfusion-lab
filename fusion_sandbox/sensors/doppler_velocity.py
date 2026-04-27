from __future__ import annotations

from typing import Any

import numpy as np

from ..math_utils import skew_symmetric
from ..trajectory import truth_at
from .faults import in_windows
from .timing import sample_times


class DopplerVelocitySensor:
    """3D velocity source with scale factor, axis misalignment, warmup, and packet loss."""

    name = "doppler_velocity"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def _outlier_offset(self, t: float) -> tuple[np.ndarray, bool]:
        for burst in self.config.get("outlier_bursts", []):
            start = float(burst.get("start_s", 0.0))
            duration = float(burst.get("duration_s", 0.0))
            if start <= t <= start + duration:
                return np.asarray(burst.get("velocity_offset_mps", [0.0, 0.0, 0.0]), dtype=float).reshape(3), True
        return np.zeros(3, dtype=float), False

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 5.0))
        noise = float(self.config.get("velocity_noise_std_mps", 0.12))
        scale = np.asarray(self.config.get("scale_factor", [1.0, 1.0, 1.0]), dtype=float).reshape(3)
        misalignment = np.asarray(self.config.get("axis_misalignment_rad", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        transform = (np.eye(3) + skew_symmetric(misalignment)) @ np.diag(scale)
        warmup = float(self.config.get("warmup_s", 0.0))
        packet_loss = float(self.config.get("packet_loss_probability", 0.0))
        jitter = float(self.config.get("timestamp_jitter_std_s", 0.0))
        windows = self.config.get("dropout_windows_s", [])
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            true_velocity = np.array([row["vx"], row["vy"], row["vz"]], dtype=float)
            offset, is_outlier = self._outlier_offset(t)
            measured = transform @ true_velocity + offset + rng.normal(0.0, noise, 3)
            warm = t < warmup
            lost = bool(packet_loss > 0.0 and rng.uniform() < packet_loss)
            in_dropout = warm or lost or in_windows(t, windows)
            availability = max(0.0, t + (float(rng.normal(0.0, jitter)) if jitter > 0.0 else 0.0))
            events.append(
                {
                    "time_s": round(availability, 10),
                    "measurement_time_s": t,
                    "sensor": "doppler_velocity",
                    "kind": "velocity_3d",
                    "valid": 0.0 if in_dropout else 1.0,
                    "is_dropout": 1.0 if in_dropout else 0.0,
                    "is_outlier": 1.0 if is_outlier else 0.0,
                    "vx": float(measured[0]),
                    "vy": float(measured[1]),
                    "vz": float(measured[2]),
                    "true_vx": float(true_velocity[0]),
                    "true_vy": float(true_velocity[1]),
                    "true_vz": float(true_velocity[2]),
                }
            )
        return events
