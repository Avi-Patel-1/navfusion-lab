from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .timing import sample_times


class WheelOdometrySensor:
    name = "wheel_odometry"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 10.0))
        noise = float(self.config.get("velocity_noise_std_mps", 0.2))
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            vel = np.array([row["vx"], row["vy"]], dtype=float) + rng.normal(0.0, noise, 2)
            events.append(
                {
                    "time_s": t,
                    "measurement_time_s": t,
                    "sensor": "wheel_odometry",
                    "kind": "planar_velocity",
                    "valid": 1.0,
                    "is_dropout": 0.0,
                    "is_outlier": 0.0,
                    "vx": float(vel[0]),
                    "vy": float(vel[1]),
                }
            )
        return events
