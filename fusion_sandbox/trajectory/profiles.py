from __future__ import annotations

import math

import numpy as np


def acceleration_for_profile(profile: str, t: float, position: np.ndarray, velocity: np.ndarray, duration_s: float) -> np.ndarray:
    if profile == "straight_line":
        return np.zeros(3, dtype=float)
    if profile == "coordinated_turn":
        turn = 0.55 * math.sin(0.16 * t) + 0.25 * math.sin(0.045 * t)
        return np.array([0.12 * math.sin(0.09 * t), turn, 0.03 * math.sin(0.18 * t)], dtype=float)
    if profile == "climb_descent":
        climb = 0.18 if t < duration_s * 0.28 else (-0.22 if t < duration_s * 0.62 else 0.06)
        return np.array([0.12 * math.sin(0.13 * t), 0.08 * math.cos(0.11 * t), climb], dtype=float)
    if profile == "aggressive_accel":
        pulse = 1.1 if 8.0 <= t <= 14.0 else (-0.9 if 24.0 <= t <= 31.0 else 0.0)
        lateral = 0.85 if 35.0 <= t <= 42.0 else 0.25 * math.sin(0.3 * t)
        return np.array([pulse + 0.18 * math.sin(0.19 * t), lateral, 0.12 * math.sin(0.23 * t)], dtype=float)
    if profile == "hover_low_speed":
        damping = -0.22 * velocity
        return damping + np.array([0.08 * math.sin(0.35 * t), 0.06 * math.cos(0.21 * t), 0.04 * math.sin(0.18 * t)], dtype=float)
    return np.array(
        [
            0.45 * math.sin(0.22 * t) + 0.12 * math.cos(0.05 * t),
            0.35 * math.cos(0.17 * t),
            0.20 * math.sin(0.31 * t) - 0.04,
        ],
        dtype=float,
    )


def figure_eight_state(t: float, duration_s: float, initial_position: np.ndarray, radius_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    period = max(duration_s * 0.65, 20.0)
    w = 2.0 * math.pi / period
    s = math.sin(w * t)
    c = math.cos(w * t)
    pos = initial_position + np.array([radius_m * s, 0.55 * radius_m * s * c, 12.0 * math.sin(0.5 * w * t)], dtype=float)
    vel = np.array(
        [
            radius_m * w * c,
            0.55 * radius_m * w * (c * c - s * s),
            6.0 * w * math.cos(0.5 * w * t),
        ],
        dtype=float,
    )
    acc = np.array(
        [
            -radius_m * w * w * s,
            -2.2 * radius_m * w * w * s * c,
            -3.0 * w * w * math.sin(0.5 * w * t),
        ],
        dtype=float,
    )
    return pos, vel, acc
