from __future__ import annotations

import math

import numpy as np

from ..math_utils import nearest_positive_definite
from .linear_kf import LinearKalmanFilter


class ParticleNavigationFilter(LinearKalmanFilter):
    """Bootstrap particle filter for nonlinear and non-Gaussian navigation studies."""

    filter_type = "particle"

    def __init__(self, *args, particle_count: int = 800, seed: int = 90210, **kwargs):
        super().__init__(*args, **kwargs)
        self.particle_count = int(particle_count)
        self.rng = np.random.default_rng(seed)
        self.particles = self.rng.multivariate_normal(self.x, self.P, self.particle_count)
        self.weights = np.full(self.particle_count, 1.0 / self.particle_count, dtype=float)
        self._sync_moments()

    def _sync_moments(self) -> None:
        self.weights = np.maximum(self.weights, 0.0)
        total = float(np.sum(self.weights))
        if total <= 0.0:
            self.weights[:] = 1.0 / self.particle_count
        else:
            self.weights /= total
        self.x = np.average(self.particles, axis=0, weights=self.weights)
        centered = self.particles - self.x
        self.P = nearest_positive_definite((centered.T * self.weights) @ centered + np.eye(9) * 1e-6)

    def _effective_sample_size(self) -> float:
        return float(1.0 / np.sum(self.weights * self.weights))

    def _roughen(self) -> None:
        span = np.ptp(self.particles, axis=0)
        kernel = self.particle_count ** (-1.0 / self.particles.shape[1])
        minimum = np.array([0.18, 0.18, 0.14, 0.04, 0.04, 0.03, 0.002, 0.002, 0.002], dtype=float)
        sigma = np.maximum(0.12 * span * kernel, minimum)
        self.particles += self.rng.normal(0.0, sigma, self.particles.shape)

    def _resample_if_needed(self) -> None:
        total = float(np.sum(self.weights))
        if total <= 0.0:
            self.weights[:] = 1.0 / self.particle_count
        else:
            self.weights /= total
        if self._effective_sample_size() > self.particle_count * 0.55:
            return
        cdf = np.cumsum(self.weights)
        cdf[-1] = 1.0
        u0 = self.rng.uniform(0.0, 1.0 / self.particle_count)
        points = u0 + np.arange(self.particle_count) / self.particle_count
        indices = np.clip(np.searchsorted(cdf, points, side="left"), 0, self.particle_count - 1)
        self.particles = self.particles[indices]
        self.weights[:] = 1.0 / self.particle_count
        self._roughen()

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        dt = float(dt)
        if dt <= 0.0:
            return
        accel = np.asarray(imu_accel, dtype=float).reshape(3)
        noise = self.rng.normal(0.0, self.profile.accel_noise * 1.5 * max(dt, 1e-9) ** 0.5, (self.particle_count, 3))
        bias_walk = self.rng.normal(0.0, self.profile.bias_walk * max(dt, 1e-9) ** 0.5, (self.particle_count, 3))
        a = accel[None, :] - self.particles[:, 6:9] + noise
        self.particles[:, :3] += self.particles[:, 3:6] * dt + 0.5 * a * dt * dt
        self.particles[:, 3:6] += a * dt
        self.particles[:, 6:9] += bias_walk
        if dropout:
            self.particles[:, :6] += self.rng.normal(0.0, self.profile.accel_noise * dt, (self.particle_count, 6))
        self.last_prediction_dt = dt
        self._sync_moments()

    def _likelihood_update(self, residuals: np.ndarray, sigma: float) -> None:
        sigma = max(float(sigma), 1e-6)
        exponent = -0.5 * np.sum((residuals / sigma) ** 2, axis=1)
        exponent -= float(np.max(exponent))
        self.weights *= np.exp(exponent)
        self._resample_if_needed()
        self._sync_moments()

    def update_gps(self, gps_xyz: np.ndarray, gps_velocity: np.ndarray | None = None, extra: dict[str, float] | None = None) -> dict[str, float]:
        z = np.asarray(gps_xyz, dtype=float).reshape(3)
        residual_before = z - self.x[:3]
        self._likelihood_update(z[None, :] - self.particles[:, :3], self.profile.gps_noise)
        if gps_velocity is not None:
            vel = np.asarray(gps_velocity, dtype=float).reshape(3)
            self._likelihood_update(vel[None, :] - self.particles[:, 3:6], self.profile.gps_velocity_noise)
        S = self.P[:3, :3] + np.eye(3) * self.profile.gps_noise**2
        nis = float(residual_before.T @ np.linalg.pinv(S) @ residual_before)
        rec = self._innovation_record("gps_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec

    def update_baro(self, altitude: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        residual_before = np.array([float(altitude) - self.x[2]], dtype=float)
        self._likelihood_update((float(altitude) - self.particles[:, 2]).reshape(-1, 1), self.profile.baro_noise)
        S = np.array([[self.P[2, 2] + self.profile.baro_noise**2]], dtype=float)
        nis = float(residual_before[0] ** 2 / max(S[0, 0], 1e-9))
        rec = self._innovation_record("barometer_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec

    def update_radar_altimeter(self, altitude_agl_m: float, terrain_altitude_m: float = 0.0, extra: dict[str, float] | None = None) -> dict[str, float]:
        altitude = float(altitude_agl_m) + float(terrain_altitude_m)
        residual_before = np.array([altitude - self.x[2]], dtype=float)
        self._likelihood_update((altitude - self.particles[:, 2]).reshape(-1, 1), self.profile.radar_altimeter_noise)
        S = np.array([[self.P[2, 2] + self.profile.radar_altimeter_noise**2]], dtype=float)
        nis = float(residual_before[0] ** 2 / max(S[0, 0], 1e-9))
        rec = self._innovation_record("radar_altimeter_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec

    def update_doppler_velocity(self, velocity_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        vel = np.asarray(velocity_xyz, dtype=float).reshape(3)
        residual_before = vel - self.x[3:6]
        self._likelihood_update(vel[None, :] - self.particles[:, 3:6], self.profile.doppler_velocity_noise)
        S = self.P[3:6, 3:6] + np.eye(3) * self.profile.doppler_velocity_noise**2
        nis = float(residual_before.T @ np.linalg.pinv(S) @ residual_before)
        rec = self._innovation_record("doppler_velocity_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec

    def update_range_beacon(self, range_m: float, beacon_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        beacon = np.asarray(beacon_xyz, dtype=float).reshape(3)
        predicted = np.linalg.norm(self.particles[:, :3] - beacon[None, :], axis=1)
        current = max(float(np.linalg.norm(self.x[:3] - beacon)), 1e-6)
        residual_before = np.array([float(range_m) - current], dtype=float)
        self._likelihood_update((float(range_m) - predicted).reshape(-1, 1), self.profile.range_noise)
        S = np.array([[self.profile.range_noise**2 + float(np.trace(self.P[:3, :3])) / 3.0]], dtype=float)
        nis = float(residual_before[0] ** 2 / max(S[0, 0], 1e-9))
        rec = self._innovation_record("range_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec

    def update_gnss_pseudorange(
        self,
        pseudorange_m: float,
        satellite_xyz: np.ndarray,
        clock_bias_correction_m: float = 0.0,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        satellite = np.asarray(satellite_xyz, dtype=float).reshape(3)
        corrected = float(pseudorange_m) - float(clock_bias_correction_m)
        predicted = np.linalg.norm(self.particles[:, :3] - satellite[None, :], axis=1)
        current = max(float(np.linalg.norm(self.x[:3] - satellite)), 1e-6)
        residual_before = np.array([corrected - current], dtype=float)
        self._likelihood_update((corrected - predicted).reshape(-1, 1), self.profile.gnss_pseudorange_noise)
        S = np.array([[self.profile.gnss_pseudorange_noise**2 + float(np.trace(self.P[:3, :3])) / 3.0]], dtype=float)
        nis = float(residual_before[0] ** 2 / max(S[0, 0], 1e-9))
        rec = self._innovation_record("gnss_pseudorange_particle", residual_before, S, nis <= self.profile.gate, nis, self.profile.gate, extra)
        rec["effective_sample_size"] = self._effective_sample_size()
        return rec
