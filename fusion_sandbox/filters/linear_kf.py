from __future__ import annotations

import numpy as np

from .base import BaseKalmanFilter


class LinearKalmanFilter(BaseKalmanFilter):
    filter_type = "linear"

    def update_gps(self, gps_xyz: np.ndarray, gps_velocity: np.ndarray | None = None, extra: dict[str, float] | None = None) -> dict[str, float]:
        if gps_velocity is None:
            H = np.zeros((3, 9))
            H[:, 0:3] = np.eye(3)
            R = np.eye(3) * self.profile.gps_noise**2
            return self.linear_update(gps_xyz, H, R, "gps_position", extra=extra)
        H = np.zeros((6, 9))
        H[0:3, 0:3] = np.eye(3)
        H[3:6, 3:6] = np.eye(3)
        z = np.concatenate([np.asarray(gps_xyz, dtype=float).reshape(3), np.asarray(gps_velocity, dtype=float).reshape(3)])
        R = np.diag([self.profile.gps_noise**2] * 3 + [self.profile.gps_velocity_noise**2] * 3)
        return self.linear_update(z, H, R, "gps_position_velocity", extra=extra)

    def update_baro(self, altitude: float, extra: dict[str, float] | None = None) -> dict[str, float]:
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        return self.linear_update(np.array([altitude]), H, np.array([[self.profile.baro_noise**2]]), "barometer_altitude", extra=extra)

    def update_wheel(self, velocity_xy: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        H = np.zeros((2, 9))
        H[:, 3:5] = np.eye(2)
        R = np.eye(2) * self.profile.wheel_noise**2
        return self.linear_update(velocity_xy, H, R, "wheel_velocity", extra=extra)

    def update_radar_altimeter(self, altitude_agl_m: float, terrain_altitude_m: float = 0.0, extra: dict[str, float] | None = None) -> dict[str, float]:
        H = np.zeros((1, 9))
        H[0, 2] = 1.0
        z = np.array([float(altitude_agl_m) + float(terrain_altitude_m)], dtype=float)
        R = np.array([[self.profile.radar_altimeter_noise**2]], dtype=float)
        return self.linear_update(z, H, R, "radar_altimeter", extra=extra)

    def update_doppler_velocity(self, velocity_xyz: np.ndarray, extra: dict[str, float] | None = None) -> dict[str, float]:
        H = np.zeros((3, 9))
        H[:, 3:6] = np.eye(3)
        R = np.eye(3) * self.profile.doppler_velocity_noise**2
        return self.linear_update(np.asarray(velocity_xyz, dtype=float).reshape(3), H, R, "doppler_velocity", extra=extra)
