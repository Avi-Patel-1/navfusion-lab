from __future__ import annotations

import math

import numpy as np

from .math_utils import wrap_angle


WGS84_RADIUS_M = 6378137.0
STANDARD_GRAVITY_MPS2 = 9.80665


def euler_to_quaternion(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=float,
    )
    return normalize_quaternion(q)


def quaternion_to_euler(q: np.ndarray) -> tuple[float, float, float]:
    w, x, y, z = normalize_quaternion(q)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1.0 else math.asin(sinp)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return roll, pitch, yaw


def normalize_quaternion(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float).reshape(4)
    norm = float(np.linalg.norm(q))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return q / norm


def quaternion_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = normalize_quaternion(q1)
    w2, x2, y2, z2 = normalize_quaternion(q2)
    return normalize_quaternion(
        np.array(
            [
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            ],
            dtype=float,
        )
    )


def quaternion_to_dcm(q: np.ndarray) -> np.ndarray:
    w, x, y, z = normalize_quaternion(q)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


def euler_to_dcm(roll: float, pitch: float, yaw: float) -> np.ndarray:
    return quaternion_to_dcm(euler_to_quaternion(roll, pitch, yaw))


def body_to_navigation(vec_body: np.ndarray, q_nav_from_body: np.ndarray) -> np.ndarray:
    return quaternion_to_dcm(q_nav_from_body) @ np.asarray(vec_body, dtype=float)


def navigation_to_body(vec_nav: np.ndarray, q_nav_from_body: np.ndarray) -> np.ndarray:
    return quaternion_to_dcm(q_nav_from_body).T @ np.asarray(vec_nav, dtype=float)


def gravity_vector(_: float = 0.0) -> np.ndarray:
    return np.array([0.0, 0.0, -STANDARD_GRAVITY_MPS2], dtype=float)


def local_tangent_from_lla(lat_rad: float, lon_rad: float, alt_m: float, reference_lla: tuple[float, float, float]) -> np.ndarray:
    ref_lat, ref_lon, ref_alt = reference_lla
    east = (lon_rad - ref_lon) * math.cos(ref_lat) * WGS84_RADIUS_M
    north = (lat_rad - ref_lat) * WGS84_RADIUS_M
    up = alt_m - ref_alt
    return np.array([east, north, up], dtype=float)


def lla_from_local_tangent(local_m: np.ndarray, reference_lla: tuple[float, float, float]) -> tuple[float, float, float]:
    ref_lat, ref_lon, ref_alt = reference_lla
    east, north, up = np.asarray(local_m, dtype=float).reshape(3)
    lat = ref_lat + north / WGS84_RADIUS_M
    lon = ref_lon + east / (math.cos(ref_lat) * WGS84_RADIUS_M)
    return float(lat), float(wrap_angle(lon)), float(ref_alt + up)
