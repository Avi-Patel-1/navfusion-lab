from __future__ import annotations

import math

import numpy as np

from ..math_utils import wrap_angle
from .linear_kf import LinearKalmanFilter


class ExtendedKalmanFilter(LinearKalmanFilter):
    filter_type = "ekf"

    def update_gps(self, gps_xyz: np.ndarray, gps_velocity: np.ndarray | None = None, extra: dict[str, float] | None = None) -> dict[str, float]:
        if self.profile.mode == "linear":
            return super().update_gps(gps_xyz, gps_velocity, extra)
        px, py, pz = self.x[:3]
        rho = max(math.sqrt(float(px * px + py * py)), 1e-6)
        h = np.array([rho, math.atan2(float(py), float(px)), pz], dtype=float)
        gps_xyz = np.asarray(gps_xyz, dtype=float).reshape(3)
        z = np.array([math.sqrt(float(gps_xyz[0] ** 2 + gps_xyz[1] ** 2)), math.atan2(float(gps_xyz[1]), float(gps_xyz[0])), gps_xyz[2]], dtype=float)
        H = np.zeros((3, 9), dtype=float)
        H[0, 0] = px / rho
        H[0, 1] = py / rho
        H[1, 0] = -py / (rho * rho)
        H[1, 1] = px / (rho * rho)
        H[2, 2] = 1.0
        R = np.diag([self.profile.gps_noise**2, (self.profile.gps_noise / max(rho, 1.0)) ** 2, self.profile.gps_noise**2])
        return self.linear_update(z - h + H @ self.x, H, R, "gps_range_bearing_altitude", wrap_indices=[1], extra=extra)

    def update_range_beacon(self, range_m: float, beacon_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        beacon = np.asarray(beacon_xyz, dtype=float).reshape(3)
        delta = self.x[:3] - beacon
        predicted = max(float(np.linalg.norm(delta)), 1e-6)
        H = np.zeros((1, 9), dtype=float)
        H[0, 0:3] = delta / predicted
        z_equiv = np.array([float(range_m) - predicted + H[0] @ self.x], dtype=float)
        R = np.array([[self.profile.range_noise**2]], dtype=float)
        return self.linear_update(z_equiv, H, R, "range_beacon", extra=extra)

    def update_gnss_pseudorange(
        self,
        pseudorange_m: float,
        satellite_xyz: np.ndarray,
        clock_bias_correction_m: float = 0.0,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        satellite = np.asarray(satellite_xyz, dtype=float).reshape(3)
        delta = self.x[:3] - satellite
        predicted = max(float(np.linalg.norm(delta)), 1e-6)
        H = np.zeros((1, 9), dtype=float)
        H[0, 0:3] = delta / predicted
        corrected_range = float(pseudorange_m) - float(clock_bias_correction_m)
        z_equiv = np.array([corrected_range - predicted + H[0] @ self.x], dtype=float)
        R = np.array([[self.profile.gnss_pseudorange_noise**2]], dtype=float)
        return self.linear_update(z_equiv, H, R, "gnss_pseudorange", extra=extra)

    def update_magnetometer(self, heading_rad: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        vx, vy = self.x[3], self.x[4]
        speed2 = max(float(vx * vx + vy * vy), 1e-6)
        predicted = math.atan2(float(vy), float(vx))
        H = np.zeros((1, 9), dtype=float)
        H[0, 3] = -vy / speed2
        H[0, 4] = vx / speed2
        innovation = float(wrap_angle(float(heading_rad) - predicted))
        z_equiv = np.array([innovation + H[0] @ self.x], dtype=float)
        R = np.array([[self.profile.mag_noise_rad**2]], dtype=float)
        return self.linear_update(z_equiv, H, R, "magnetometer_heading", wrap_indices=[0], gate=self.profile.gate * 0.5, extra=extra)
