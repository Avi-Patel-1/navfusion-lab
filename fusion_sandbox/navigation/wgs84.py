from __future__ import annotations

import math

import numpy as np


WGS84_A_M = 6378137.0
WGS84_F = 1.0 / 298.257223563
WGS84_B_M = WGS84_A_M * (1.0 - WGS84_F)
WGS84_E2 = WGS84_F * (2.0 - WGS84_F)
EARTH_RATE_RADPS = 7.292115e-5


def geodetic_to_ecef(lat_rad: float, lon_rad: float, alt_m: float) -> np.ndarray:
    """Convert geodetic latitude, longitude, altitude to ECEF meters."""
    sin_lat = math.sin(lat_rad)
    cos_lat = math.cos(lat_rad)
    sin_lon = math.sin(lon_rad)
    cos_lon = math.cos(lon_rad)
    n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
    x = (n + alt_m) * cos_lat * cos_lon
    y = (n + alt_m) * cos_lat * sin_lon
    z = (n * (1.0 - WGS84_E2) + alt_m) * sin_lat
    return np.array([x, y, z], dtype=float)


def ecef_to_geodetic(ecef_m: np.ndarray, tolerance: float = 1e-12) -> tuple[float, float, float]:
    """Convert ECEF meters to geodetic latitude, longitude, altitude."""
    x, y, z = np.asarray(ecef_m, dtype=float).reshape(3)
    lon = math.atan2(y, x)
    p = math.hypot(x, y)
    lat = math.atan2(z, p * (1.0 - WGS84_E2))
    alt = 0.0
    for _ in range(12):
        sin_lat = math.sin(lat)
        n = WGS84_A_M / math.sqrt(1.0 - WGS84_E2 * sin_lat * sin_lat)
        alt_new = p / max(math.cos(lat), 1e-15) - n
        lat_new = math.atan2(z, p * (1.0 - WGS84_E2 * n / (n + alt_new)))
        if abs(lat_new - lat) < tolerance and abs(alt_new - alt) < 1e-8:
            lat, alt = lat_new, alt_new
            break
        lat, alt = lat_new, alt_new
    return float(lat), float(lon), float(alt)


def _ecef_to_enu_matrix(lat_rad: float, lon_rad: float) -> np.ndarray:
    s_lat, c_lat = math.sin(lat_rad), math.cos(lat_rad)
    s_lon, c_lon = math.sin(lon_rad), math.cos(lon_rad)
    return np.array(
        [
            [-s_lon, c_lon, 0.0],
            [-s_lat * c_lon, -s_lat * s_lon, c_lat],
            [c_lat * c_lon, c_lat * s_lon, s_lat],
        ],
        dtype=float,
    )


def ecef_to_enu(ecef_m: np.ndarray, reference_lla_rad_m: tuple[float, float, float]) -> np.ndarray:
    ref = geodetic_to_ecef(*reference_lla_rad_m)
    return _ecef_to_enu_matrix(reference_lla_rad_m[0], reference_lla_rad_m[1]) @ (np.asarray(ecef_m, dtype=float).reshape(3) - ref)


def enu_to_ecef(enu_m: np.ndarray, reference_lla_rad_m: tuple[float, float, float]) -> np.ndarray:
    ref = geodetic_to_ecef(*reference_lla_rad_m)
    return ref + _ecef_to_enu_matrix(reference_lla_rad_m[0], reference_lla_rad_m[1]).T @ np.asarray(enu_m, dtype=float).reshape(3)


def ecef_to_ned(ecef_m: np.ndarray, reference_lla_rad_m: tuple[float, float, float]) -> np.ndarray:
    enu = ecef_to_enu(ecef_m, reference_lla_rad_m)
    return np.array([enu[1], enu[0], -enu[2]], dtype=float)


def ned_to_ecef(ned_m: np.ndarray, reference_lla_rad_m: tuple[float, float, float]) -> np.ndarray:
    ned = np.asarray(ned_m, dtype=float).reshape(3)
    return enu_to_ecef(np.array([ned[1], ned[0], -ned[2]], dtype=float), reference_lla_rad_m)


def gravity_normal(lat_rad: float, alt_m: float = 0.0) -> float:
    """Somigliana normal gravity in m/s^2 with a first-order altitude correction."""
    sin_lat = math.sin(lat_rad)
    sin2 = sin_lat * sin_lat
    gamma_equator = 9.7803253359
    k = 0.00193185265241
    gamma = gamma_equator * (1.0 + k * sin2) / math.sqrt(1.0 - WGS84_E2 * sin2)
    return float(gamma - 3.086e-6 * alt_m)
