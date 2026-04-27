from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..fusion.gating import gate_measurement
from ..fusion.measurement_update import joseph_update
from ..fusion.time_update import inflate_covariance
from ..math_utils import nearest_positive_definite, safe_inverse, wrap_angle


@dataclass(frozen=True)
class FilterProfile:
    name: str
    accel_noise: float
    bias_walk: float
    gps_noise: float
    gps_velocity_noise: float
    baro_noise: float
    mag_noise_rad: float
    range_noise: float
    wheel_noise: float
    gate: float
    dropout_inflation: float
    mode: str = "ekf"
    radar_altimeter_noise: float = 0.6
    doppler_velocity_noise: float = 0.18
    gnss_pseudorange_noise: float = 1.8


PROFILE_MAP: dict[str, FilterProfile] = {
    "aggressive": FilterProfile("aggressive", 0.08, 0.0005, 1.7, 0.18, 0.45, 0.04, 0.7, 0.14, 35.0, 1.0005, "ekf"),
    "nominal": FilterProfile("nominal", 0.18, 0.0015, 2.4, 0.28, 0.75, 0.055, 1.0, 0.22, 55.0, 1.0015, "ekf"),
    "conservative": FilterProfile("conservative", 0.35, 0.0040, 3.2, 0.42, 1.20, 0.075, 1.6, 0.35, 80.0, 1.0030, "linear"),
    "dropout_robust": FilterProfile("dropout_robust", 0.42, 0.0055, 3.0, 0.36, 1.05, 0.075, 1.4, 0.30, 70.0, 1.0080, "ekf"),
    "high_bias": FilterProfile("high_bias", 0.28, 0.0120, 2.8, 0.34, 0.95, 0.065, 1.3, 0.28, 65.0, 1.0050, "ekf"),
}


PROFILES = list(PROFILE_MAP.values())


class BaseKalmanFilter:
    filter_type = "base"

    def __init__(self, profile: FilterProfile, initial_position: np.ndarray, initial_velocity: np.ndarray | None = None):
        self.profile = profile
        self.x = np.zeros(9, dtype=float)
        self.x[:3] = np.asarray(initial_position, dtype=float).reshape(3)
        if initial_velocity is not None:
            self.x[3:6] = np.asarray(initial_velocity, dtype=float).reshape(3)
        self.P = np.diag([35.0, 35.0, 35.0, 12.0, 12.0, 12.0, 0.15, 0.15, 0.15]).astype(float)
        self.last_prediction_dt = 0.0

    @property
    def yaw_estimate(self) -> float:
        vx, vy = self.x[3], self.x[4]
        return math.atan2(float(vy), float(vx)) if vx * vx + vy * vy > 1e-12 else 0.0

    def process_noise(self, dt: float) -> np.ndarray:
        q_acc = self.profile.accel_noise**2
        q_bias = self.profile.bias_walk**2
        return np.diag([0.25 * q_acc * dt**4] * 3 + [q_acc * dt**2] * 3 + [q_bias * max(dt, 1e-9)] * 3)

    def transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = -0.5 * np.eye(3) * dt * dt
        F[3:6, 6:9] = -np.eye(3) * dt
        return F

    def propagate_state(self, state: np.ndarray, imu_accel: np.ndarray, dt: float) -> np.ndarray:
        x = state.copy()
        a = np.asarray(imu_accel, dtype=float).reshape(3) - x[6:9]
        x[:3] = x[:3] + x[3:6] * dt + 0.5 * a * dt * dt
        x[3:6] = x[3:6] + a * dt
        return x

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        F = self.transition_matrix(dt)
        self.x = self.propagate_state(self.x, imu_accel, dt)
        self.P = F @ self.P @ F.T + self.process_noise(dt)
        if dropout:
            self.P = inflate_covariance(self.P, self.profile.dropout_inflation)
        self.P = nearest_positive_definite(self.P)
        self.last_prediction_dt = dt

    def _innovation_record(
        self,
        source: str,
        residual: np.ndarray,
        S: np.ndarray,
        accepted: bool,
        nis: float,
        gate: float,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        y = np.asarray(residual, dtype=float).reshape(-1)
        record: dict[str, float] = {
            "source": source,
            "nis": float(nis),
            "gate": float(gate),
            "accepted": 1.0 if accepted else 0.0,
            "residual_norm": float(np.linalg.norm(y)),
            "innovation_std": float(math.sqrt(max(np.trace(S), 0.0))),
        }
        for i, value in enumerate(y[:6]):
            record[f"residual_{i}"] = float(value)
        if extra:
            record.update(extra)
        return record

    def linear_update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        source: str,
        wrap_indices: list[int] | None = None,
        gate: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        z = np.asarray(z, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        residual = z - H @ self.x
        for idx in wrap_indices or []:
            residual[idx] = float(wrap_angle(residual[idx]))
        S = H @ self.P @ H.T + R
        threshold = float(self.profile.gate if gate is None else gate)
        accepted, nis = gate_measurement(residual, S, threshold)
        if accepted:
            self.x, self.P, _ = joseph_update(self.x, self.P, H, R, residual)
        return self._innovation_record(source, residual, S, accepted, nis, threshold, extra)

    def covariance_trace(self) -> float:
        return float(np.trace(self.P))

    def position_sigma(self) -> float:
        return float(math.sqrt(max(np.trace(self.P[:3, :3]), 0.0)))

    def velocity_sigma(self) -> float:
        return float(math.sqrt(max(np.trace(self.P[3:6, 3:6]), 0.0)))

    def nees_proxy(self, position_error: np.ndarray) -> float:
        Ppos = self.P[:3, :3] + np.eye(3) * 1e-9
        return float(position_error.T @ safe_inverse(Ppos) @ position_error)
