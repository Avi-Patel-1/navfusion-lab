from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .timing import sample_times


class RangeBeaconSensor:
    name = "range_beacon"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", False):
            return []
        rate = float(self.config.get("rate_hz", 2.0))
        noise = float(self.config.get("range_noise_std_m", 0.9))
        beacons = [np.array(b, dtype=float) for b in self.config.get("beacons_m", [])]
        events: list[dict[str, float]] = []
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            pos = np.array([row["px"], row["py"], row["pz"]], dtype=float)
            for index, beacon in enumerate(beacons):
                true_range = float(np.linalg.norm(pos - beacon))
                events.append(
                    {
                        "time_s": t,
                        "measurement_time_s": t,
                        "sensor": "range_beacon",
                        "kind": "range",
                        "valid": 1.0,
                        "is_dropout": 0.0,
                        "is_outlier": 0.0,
                        "beacon_id": float(index),
                        "beacon_x": float(beacon[0]),
                        "beacon_y": float(beacon[1]),
                        "beacon_z": float(beacon[2]),
                        "range_m": true_range + float(rng.normal(0.0, noise)),
                    }
                )
        return events
