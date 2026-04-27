from __future__ import annotations

from typing import Any

import numpy as np

from ..math_utils import wrap_angle
from ..trajectory import truth_at
from .timing import sample_times


class MagnetometerSensor:
    name = "magnetometer"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 2.0))
        noise = float(self.config.get("heading_noise_std_rad", 0.035))
        disturbances = self.config.get("disturbances", [])
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            offset = 0.0
            disturbed = False
            for event in disturbances:
                start = float(event.get("start_s", 0.0))
                duration = float(event.get("duration_s", 0.0))
                if start <= t <= start + duration:
                    offset += float(event.get("heading_offset_rad", 0.0))
                    disturbed = True
            heading = float(wrap_angle(float(row["yaw_rad"]) + offset + rng.normal(0.0, noise)))
            events.append(
                {
                    "time_s": t,
                    "measurement_time_s": t,
                    "sensor": "magnetometer",
                    "kind": "heading",
                    "valid": 1.0,
                    "is_dropout": 0.0,
                    "is_outlier": 1.0 if disturbed else 0.0,
                    "heading_rad": heading,
                    "disturbance_rad": offset,
                }
            )
        return events
