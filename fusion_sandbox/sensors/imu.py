from __future__ import annotations

from typing import Any

import numpy as np

from ..trajectory import truth_at
from .timing import sample_times


class IMUSensor:
    name = "imu"

    def __init__(self, config: dict[str, Any]):
        self.config = config

    def generate(self, truth: list[dict[str, float]], rng: np.random.Generator) -> list[dict[str, float]]:
        if not self.config.get("enabled", True):
            return []
        rate = float(self.config.get("rate_hz", 20.0))
        accel_noise = float(self.config.get("accel_noise_std_mps2", 0.045))
        gyro_noise = float(self.config.get("gyro_noise_std_radps", 0.002))
        bias = np.array(self.config.get("accel_bias_mps2", [0.0, 0.0, 0.0]), dtype=float)
        gyro_bias = np.array(self.config.get("gyro_bias_radps", [0.0, 0.0, 0.0]), dtype=float)
        scale = np.array(self.config.get("scale_factor", [1.0, 1.0, 1.0]), dtype=float)
        walk = float(self.config.get("accel_bias_walk_std_mps2_sqrt_s", 0.0))
        events: list[dict[str, float]] = []
        previous_t = 0.0
        previous_yaw = float(truth[0].get("yaw_rad", 0.0))
        for t in sample_times(float(truth[-1]["time_s"]), rate):
            row = truth_at(truth, t)
            dt = max(t - previous_t, 0.0)
            if dt > 0.0 and walk > 0.0:
                bias = bias + rng.normal(0.0, walk * dt**0.5, 3)
            accel_true = np.array([row["ax"], row["ay"], row["az"]], dtype=float)
            accel = scale * accel_true + bias + rng.normal(0.0, accel_noise, 3)
            yaw = float(row.get("yaw_rad", previous_yaw))
            yaw_rate = (yaw - previous_yaw) / dt if dt > 1e-9 else 0.0
            gyro = np.array([0.0, 0.0, yaw_rate], dtype=float) + gyro_bias + rng.normal(0.0, gyro_noise, 3)
            events.append(
                {
                    "time_s": t,
                    "measurement_time_s": t,
                    "sensor": "imu",
                    "kind": "inertial",
                    "valid": 1.0,
                    "is_dropout": 0.0,
                    "is_outlier": 0.0,
                    "ax": float(accel[0]),
                    "ay": float(accel[1]),
                    "az": float(accel[2]),
                    "gx": float(gyro[0]),
                    "gy": float(gyro[1]),
                    "gz": float(gyro[2]),
                    "bax_true": float(bias[0]),
                    "bay_true": float(bias[1]),
                    "baz_true": float(bias[2]),
                }
            )
            previous_t = t
            previous_yaw = yaw
        return events
