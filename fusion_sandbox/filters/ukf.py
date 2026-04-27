from __future__ import annotations

import numpy as np

from ..fusion.gating import gate_measurement
from ..math_utils import nearest_positive_definite, safe_inverse, wrap_angle
from .base import BaseKalmanFilter


class UnscentedKalmanFilter(BaseKalmanFilter):
    filter_type = "ukf"

    def __init__(self, *args, alpha: float = 0.35, beta: float = 2.0, kappa: float = 0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

    def _weights(self) -> tuple[np.ndarray, np.ndarray, float]:
        n = self.x.size
        lam = self.alpha**2 * (n + self.kappa) - n
        wm = np.full(2 * n + 1, 1.0 / (2.0 * (n + lam)), dtype=float)
        wc = wm.copy()
        wm[0] = lam / (n + lam)
        wc[0] = wm[0] + (1.0 - self.alpha**2 + self.beta)
        return wm, wc, lam

    def sigma_points(self) -> np.ndarray:
        n = self.x.size
        _, _, lam = self._weights()
        scale = n + lam
        jitter = 1e-9
        for _ in range(5):
            try:
                root = np.linalg.cholesky(nearest_positive_definite(self.P + np.eye(n) * jitter) * scale)
                break
            except np.linalg.LinAlgError:
                jitter *= 10.0
        else:
            root = np.linalg.cholesky(np.eye(n) * scale)
        points = [self.x]
        for i in range(n):
            points.append(self.x + root[:, i])
            points.append(self.x - root[:, i])
        return np.vstack(points)

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        wm, wc, _ = self._weights()
        propagated = np.vstack([self.propagate_state(point, imu_accel, dt) for point in self.sigma_points()])
        x_pred = np.sum(propagated * wm[:, None], axis=0)
        P_pred = self.process_noise(dt)
        for weight, point in zip(wc, propagated):
            diff = point - x_pred
            P_pred += weight * np.outer(diff, diff)
        self.x = x_pred
        self.P = nearest_positive_definite(P_pred * (self.profile.dropout_inflation if dropout else 1.0))

    def _unscented_update(
        self,
        z: np.ndarray,
        measurement_fn,
        R: np.ndarray,
        source: str,
        angle_indices: list[int] | None = None,
        gate: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        z = np.asarray(z, dtype=float).reshape(-1)
        R = np.asarray(R, dtype=float)
        wm, wc, _ = self._weights()
        sigmas = self.sigma_points()
        zsig = np.vstack([measurement_fn(point) for point in sigmas])
        z_pred = np.sum(zsig * wm[:, None], axis=0)
        for idx in angle_indices or []:
            s = float(np.sum(np.sin(zsig[:, idx]) * wm))
            c = float(np.sum(np.cos(zsig[:, idx]) * wm))
            z_pred[idx] = np.arctan2(s, c)
        S = R.copy()
        Pxz = np.zeros((self.x.size, z.size), dtype=float)
        for weight, point, zpoint in zip(wc, sigmas, zsig):
            dx = point - self.x
            dz = zpoint - z_pred
            for idx in angle_indices or []:
                dz[idx] = float(wrap_angle(dz[idx]))
            S += weight * np.outer(dz, dz)
            Pxz += weight * np.outer(dx, dz)
        residual = z - z_pred
        for idx in angle_indices or []:
            residual[idx] = float(wrap_angle(residual[idx]))
        threshold = float(self.profile.gate if gate is None else gate)
        accepted, nis = gate_measurement(residual, S, threshold)
        if accepted:
            K = Pxz @ safe_inverse(S)
            self.x = self.x + K @ residual
            self.P = nearest_positive_definite(self.P - K @ S @ K.T)
        return self._innovation_record(source, residual, S, accepted, nis, threshold, extra)

    def update_gps(self, gps_xyz: np.ndarray, gps_velocity: np.ndarray | None = None, extra: dict[str, float] | None = None) -> dict[str, float]:
        gps_xyz = np.asarray(gps_xyz, dtype=float).reshape(3)
        R = np.diag([self.profile.gps_noise**2] * 3)
        return self._unscented_update(gps_xyz, lambda x: x[:3], R, "gps_position_ukf", extra=extra)

    def update_baro(self, altitude: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        return self._unscented_update(np.array([altitude]), lambda x: np.array([x[2]]), np.array([[self.profile.baro_noise**2]]), "barometer_altitude", extra=extra)

    def update_range_beacon(self, range_m: float, beacon_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        beacon = np.asarray(beacon_xyz, dtype=float).reshape(3)
        return self._unscented_update(
            np.array([range_m]),
            lambda x: np.array([np.linalg.norm(x[:3] - beacon)]),
            np.array([[self.profile.range_noise**2]]),
            "range_beacon_ukf",
            extra=extra,
        )

    def update_magnetometer(self, heading_rad: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        return self._unscented_update(
            np.array([heading_rad]),
            lambda x: np.array([np.arctan2(x[4], x[3])]),
            np.array([[self.profile.mag_noise_rad**2]]),
            "magnetometer_heading_ukf",
            angle_indices=[0],
            gate=self.profile.gate * 0.5,
            extra=extra,
        )

    def update_wheel(self, velocity_xy: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        return self._unscented_update(
            np.asarray(velocity_xy, dtype=float).reshape(2),
            lambda x: x[3:5],
            np.eye(2) * self.profile.wheel_noise**2,
            "wheel_velocity",
            extra=extra,
        )
