from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite
from .linear_kf import LinearKalmanFilter


class AlphaBetaGammaFilter(LinearKalmanFilter):
    """Deterministic alpha-beta-gamma baseline for position-aided motion tracking.

    The sandbox state vector keeps nine slots for compatibility with the Kalman
    filters. For this baseline, the final three slots are interpreted as a
    constant acceleration estimate rather than accelerometer bias.
    """

    filter_type = "abg"

    def __init__(self, *args, alpha: float = 0.65, beta: float = 0.20, gamma: float = 0.03, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self._last_position_update_time: float | None = None
        self._last_altitude_update_time: float | None = None

    def transition_matrix(self, dt: float) -> np.ndarray:
        F = np.eye(9)
        F[0:3, 3:6] = np.eye(3) * dt
        F[0:3, 6:9] = 0.5 * np.eye(3) * dt * dt
        F[3:6, 6:9] = np.eye(3) * dt
        return F

    def process_noise(self, dt: float) -> np.ndarray:
        accel_drive = max(self.profile.accel_noise, 1e-6) ** 2
        accel_state = max(self.profile.bias_walk, 1e-6) ** 2
        return np.diag([0.25 * accel_drive * dt**4] * 3 + [accel_drive * dt**2] * 3 + [accel_state * max(dt, 1e-9)] * 3)

    def propagate_state(self, state: np.ndarray, imu_accel: np.ndarray, dt: float) -> np.ndarray:
        x = state.copy()
        accel = x[6:9]
        x[:3] = x[:3] + x[3:6] * dt + 0.5 * accel * dt * dt
        x[3:6] = x[3:6] + accel * dt
        return x

    def _measurement_dt(self, extra: dict[str, float] | None, attr_name: str) -> float:
        fallback = max(self.last_prediction_dt, 1e-3)
        if extra is None or "time_s" not in extra:
            return fallback
        t = float(extra["time_s"])
        previous = getattr(self, attr_name)
        setattr(self, attr_name, t)
        if previous is None:
            return 1.0
        return max(t - float(previous), fallback, 1e-3)

    @staticmethod
    def _clamp_vector(vec: np.ndarray, limit: float) -> np.ndarray:
        norm = float(np.linalg.norm(vec))
        if norm <= limit or norm <= 1e-12:
            return vec
        return vec * (limit / norm)

    def _bound_covariance(self) -> None:
        caps = np.array([120.0, 120.0, 120.0, 30.0, 30.0, 30.0, 6.0, 6.0, 6.0], dtype=float)
        diag = np.clip(np.diag(self.P), 1e-6, caps)
        self.P = np.diag(diag)

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        super().predict(imu_accel, dt, dropout)
        self._bound_covariance()

    def update_gps(self, gps_xyz: np.ndarray, gps_velocity: np.ndarray | None = None, extra: dict[str, float] | None = None) -> dict[str, float]:
        z = np.asarray(gps_xyz, dtype=float).reshape(3)
        residual = z - self.x[:3]
        dt = self._measurement_dt(extra, "_last_position_update_time")
        S = self.P[:3, :3] + np.eye(3) * self.profile.gps_noise**2
        nis = float(residual.T @ np.linalg.pinv(S) @ residual)
        accepted = nis <= self.profile.gate * 2.0
        if not accepted:
            return self._innovation_record("gps_alpha_beta_gamma", residual, S, False, nis, self.profile.gate * 2.0, extra)
        self.x[:3] += self.alpha * residual
        self.x[3:6] += (self.beta / dt) * residual
        self.x[6:9] += self._clamp_vector((2.0 * self.gamma / (dt * dt)) * residual, 3.0)
        if gps_velocity is not None:
            self.x[3:6] = 0.65 * self.x[3:6] + 0.35 * np.asarray(gps_velocity, dtype=float).reshape(3)
        self.P[:3, :3] *= max(1.0 - self.alpha * 0.25, 0.2)
        self.P[3:6, 3:6] *= max(1.0 - self.beta * 0.25, 0.35)
        self.P = nearest_positive_definite(self.P)
        self._bound_covariance()
        record = self._innovation_record("gps_alpha_beta_gamma", residual, S, True, nis, self.profile.gate * 2.0, extra)
        record.update({"alpha": self.alpha, "beta": self.beta, "gamma": self.gamma})
        return record

    def update_baro(self, altitude: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        residual = np.array([float(altitude) - self.x[2]], dtype=float)
        dt = self._measurement_dt(extra, "_last_altitude_update_time")
        self.x[2] += self.alpha * residual[0] * 0.45
        self.x[5] += self.beta * residual[0] * 0.45 / dt
        self.x[8] += float(np.clip(2.0 * self.gamma * residual[0] * 0.45 / (dt * dt), -1.5, 1.5))
        self.P[2, 2] *= 0.85
        self.P = nearest_positive_definite(self.P)
        self._bound_covariance()
        S = np.array([[self.P[2, 2] + self.profile.baro_noise**2]], dtype=float)
        return self._innovation_record("barometer_alpha_beta_gamma", residual, S, True, float(residual[0] ** 2 / max(S[0, 0], 1e-9)), self.profile.gate, extra)

    def update_range_beacon(self, range_m: float, beacon_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        beacon = np.asarray(beacon_xyz, dtype=float).reshape(3)
        delta = self.x[:3] - beacon
        predicted = max(float(np.linalg.norm(delta)), 1e-6)
        los = delta / predicted
        residual = np.array([float(range_m) - predicted], dtype=float)
        S = np.array([[self.profile.range_noise**2 + float(np.trace(self.P[:3, :3])) / 3.0]], dtype=float)
        nis = float(residual[0] ** 2 / max(S[0, 0], 1e-9))
        accepted = nis <= self.profile.gate * 4.0
        if accepted:
            correction = self.alpha * residual[0] * los
            self.x[:3] += correction
            self.x[3:6] += self.beta * correction / max(self.last_prediction_dt, 1e-3)
            self.P[:3, :3] *= 0.92
            self.P = nearest_positive_definite(self.P)
            self._bound_covariance()
        return self._innovation_record("range_alpha_beta_gamma", residual, S, accepted, nis, self.profile.gate * 4.0, extra)
