from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .ins import coning_correction, sculling_correction


@dataclass
class PreintegratedIMU:
    delta_theta_rad: np.ndarray
    delta_velocity_mps: np.ndarray
    delta_position_m: np.ndarray
    dt_s: float
    sample_count: int


class IMUPreintegrator:
    """Accumulates IMU increments for delayed or factor-graph style updates."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.delta_theta_rad = np.zeros(3, dtype=float)
        self.delta_velocity_mps = np.zeros(3, dtype=float)
        self.delta_position_m = np.zeros(3, dtype=float)
        self.dt_s = 0.0
        self.sample_count = 0
        self._last_delta_theta: np.ndarray | None = None
        self._last_delta_v: np.ndarray | None = None

    def add_sample(self, accel_body_mps2: np.ndarray, gyro_body_radps: np.ndarray, dt_s: float) -> None:
        dt = float(dt_s)
        dtheta = np.asarray(gyro_body_radps, dtype=float).reshape(3) * dt
        dv = np.asarray(accel_body_mps2, dtype=float).reshape(3) * dt
        if self._last_delta_theta is not None and self._last_delta_v is not None:
            dtheta_total = coning_correction(self._last_delta_theta, dtheta) - self._last_delta_theta
            dv_total = sculling_correction(self._last_delta_v, dv, self._last_delta_theta, dtheta) - self._last_delta_v
        else:
            dtheta_total = dtheta
            dv_total = dv
        self.delta_position_m += self.delta_velocity_mps * dt + 0.5 * dv_total * dt
        self.delta_velocity_mps += dv_total
        self.delta_theta_rad += dtheta_total
        self.dt_s += dt
        self.sample_count += 1
        self._last_delta_theta = dtheta
        self._last_delta_v = dv

    def result(self) -> PreintegratedIMU:
        return PreintegratedIMU(
            self.delta_theta_rad.copy(),
            self.delta_velocity_mps.copy(),
            self.delta_position_m.copy(),
            float(self.dt_s),
            int(self.sample_count),
        )
