from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .timing import sample_times


class BarometerSensor:
    name = "barometer"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", True):
            return []
        rate = float(self.config.get("rate_hz", 5.0))
        noise = float(self.config.get("altitude_noise_std_m", 0.7))
        bias = float(self.config.get("bias_m", 0.0))
        walk = float(self.config.get("bias_walk_std_m_sqrt_s", 0.0))
        events: list[dict[str, float]] = []
        previous_t = 0.0
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            dt = max(t - previous_t, 0.0)
            if dt > 0.0 and walk > 0.0:
                bias += float(rng.normal(0.0, walk * dt**0.5))
            altitude = float(row["pz"]) + bias + float(rng.normal(0.0, noise))
            events.append(
                {
                    "time_s": t,
                    "measurement_time_s": t,
                    "sensor": "barometer",
                    "kind": "altitude",
                    "valid": 1.0,
                    "is_dropout": 0.0,
                    "is_outlier": 0.0,
                    "altitude_m": altitude,
                    "bias_true_m": bias,
                }
            )
            previous_t = t
        return events
