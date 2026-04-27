from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import normalize_quaternion, quaternion_multiply, quaternion_to_dcm
from ..math_utils import skew_symmetric


@dataclass
class StrapdownState:
    position_m: np.ndarray
    velocity_mps: np.ndarray
    q_nav_from_body: np.ndarray
    gyro_bias_radps: np.ndarray
    accel_bias_mps2: np.ndarray

    def copy(self) -> "StrapdownState":
        return StrapdownState(
            self.position_m.copy(),
            self.velocity_mps.copy(),
            self.q_nav_from_body.copy(),
            self.gyro_bias_radps.copy(),
            self.accel_bias_mps2.copy(),
        )


def small_angle_quaternion(delta_rad: np.ndarray) -> np.ndarray:
    delta = np.asarray(delta_rad, dtype=float).reshape(3)
    angle = float(np.linalg.norm(delta))
    if angle < 1e-12:
        return normalize_quaternion(np.array([1.0, 0.5 * delta[0], 0.5 * delta[1], 0.5 * delta[2]], dtype=float))
    axis = delta / angle
    return normalize_quaternion(np.r_[np.cos(angle * 0.5), axis * np.sin(angle * 0.5)])


def propagate_quaternion(q_nav_from_body: np.ndarray, gyro_radps: np.ndarray, dt_s: float) -> np.ndarray:
    """Integrate body angular rate into a navigation-from-body quaternion."""
    dq = small_angle_quaternion(np.asarray(gyro_radps, dtype=float).reshape(3) * float(dt_s))
    return normalize_quaternion(quaternion_multiply(q_nav_from_body, dq))


def coning_correction(delta_theta_1: np.ndarray, delta_theta_2: np.ndarray) -> np.ndarray:
    return np.asarray(delta_theta_1, dtype=float) + np.asarray(delta_theta_2, dtype=float) + (2.0 / 3.0) * np.cross(delta_theta_1, delta_theta_2)


def sculling_correction(delta_v_1: np.ndarray, delta_v_2: np.ndarray, delta_theta_1: np.ndarray, delta_theta_2: np.ndarray) -> np.ndarray:
    return (
        np.asarray(delta_v_1, dtype=float)
        + np.asarray(delta_v_2, dtype=float)
        + 0.5 * (np.cross(delta_theta_1, delta_v_2) + np.cross(delta_theta_2, delta_v_1))
        + (2.0 / 3.0) * (np.cross(delta_theta_1, delta_v_1) + np.cross(delta_theta_2, delta_v_2))
    )


def mechanize(
    state: StrapdownState,
    accel_body_mps2: np.ndarray,
    gyro_body_radps: np.ndarray,
    dt_s: float,
    gravity_nav_mps2: np.ndarray | None = None,
) -> StrapdownState:
    """One strapdown INS mechanization step in a local navigation frame."""
    dt = float(dt_s)
    next_state = state.copy()
    corrected_gyro = np.asarray(gyro_body_radps, dtype=float).reshape(3) - next_state.gyro_bias_radps
    corrected_accel = np.asarray(accel_body_mps2, dtype=float).reshape(3) - next_state.accel_bias_mps2
    q_next = propagate_quaternion(next_state.q_nav_from_body, corrected_gyro, dt)
    dcm_mid = quaternion_to_dcm(normalize_quaternion(next_state.q_nav_from_body + q_next))
    gravity = np.array([0.0, 0.0, -9.80665], dtype=float) if gravity_nav_mps2 is None else np.asarray(gravity_nav_mps2, dtype=float).reshape(3)
    accel_nav = dcm_mid @ corrected_accel + gravity
    next_state.position_m = next_state.position_m + next_state.velocity_mps * dt + 0.5 * accel_nav * dt * dt
    next_state.velocity_mps = next_state.velocity_mps + accel_nav * dt
    next_state.q_nav_from_body = q_next
    return next_state


def attitude_error_jacobian(specific_force_body: np.ndarray, q_nav_from_body: np.ndarray) -> np.ndarray:
    return -quaternion_to_dcm(q_nav_from_body) @ skew_symmetric(np.asarray(specific_force_body, dtype=float).reshape(3))
