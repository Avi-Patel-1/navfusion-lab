from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..frames import navigation_to_body, normalize_quaternion, quaternion_multiply, quaternion_to_euler
from .ins import propagate_quaternion, small_angle_quaternion


def _unit_or_none(vec: np.ndarray, min_norm: float = 1e-9) -> np.ndarray | None:
    arr = np.asarray(vec, dtype=float).reshape(3)
    norm = float(np.linalg.norm(arr))
    if norm < min_norm:
        return None
    return arr / norm


@dataclass
class AttitudeEstimate:
    q_nav_from_body: np.ndarray
    roll_rad: float
    pitch_rad: float
    yaw_rad: float


class ComplementaryAttitudeFilter:
    """Quaternion complementary attitude filter using gyro prediction and vector correction."""

    def __init__(self, q_nav_from_body: np.ndarray | None = None, accel_gain: float = 0.04, mag_gain: float = 0.02):
        self.q = normalize_quaternion(np.array([1.0, 0.0, 0.0, 0.0]) if q_nav_from_body is None else q_nav_from_body)
        self.accel_gain = float(accel_gain)
        self.mag_gain = float(mag_gain)

    def predict(self, gyro_body_radps: np.ndarray, dt_s: float) -> None:
        self.q = propagate_quaternion(self.q, gyro_body_radps, float(dt_s))

    def _apply_body_vector_correction(self, measured_body: np.ndarray, reference_nav: np.ndarray, gain: float) -> float:
        measured = _unit_or_none(measured_body)
        reference = _unit_or_none(reference_nav)
        if measured is None or reference is None or gain <= 0.0:
            return 0.0
        expected_body = navigation_to_body(reference, self.q)
        error_body = np.cross(measured, expected_body)
        self.q = normalize_quaternion(quaternion_multiply(self.q, small_angle_quaternion(gain * error_body)))
        return float(np.linalg.norm(error_body))

    def update_accel(self, accel_body_mps2: np.ndarray) -> float:
        return self._apply_body_vector_correction(accel_body_mps2, np.array([0.0, 0.0, 1.0]), self.accel_gain)

    def update_mag(self, mag_body: np.ndarray, reference_nav: np.ndarray | None = None) -> float:
        ref = np.array([1.0, 0.0, 0.0], dtype=float) if reference_nav is None else np.asarray(reference_nav, dtype=float)
        return self._apply_body_vector_correction(mag_body, ref, self.mag_gain)

    def update(self, gyro_body_radps: np.ndarray, accel_body_mps2: np.ndarray | None, dt_s: float, mag_body: np.ndarray | None = None) -> AttitudeEstimate:
        self.predict(gyro_body_radps, dt_s)
        if accel_body_mps2 is not None:
            self.update_accel(accel_body_mps2)
        if mag_body is not None:
            self.update_mag(mag_body)
        return self.estimate()

    def estimate(self) -> AttitudeEstimate:
        roll, pitch, yaw = quaternion_to_euler(self.q)
        return AttitudeEstimate(self.q.copy(), roll, pitch, yaw)


class MahonyAttitudeFilter(ComplementaryAttitudeFilter):
    """Mahony-style nonlinear complementary filter with integral gyro-bias correction."""

    def __init__(
        self,
        q_nav_from_body: np.ndarray | None = None,
        proportional_gain: float = 0.8,
        integral_gain: float = 0.04,
    ):
        super().__init__(q_nav_from_body=q_nav_from_body, accel_gain=0.0, mag_gain=0.0)
        self.proportional_gain = float(proportional_gain)
        self.integral_gain = float(integral_gain)
        self.integral_error = np.zeros(3, dtype=float)
        self.gyro_bias_radps = np.zeros(3, dtype=float)

    def _innovation(self, accel_body_mps2: np.ndarray | None, mag_body: np.ndarray | None = None) -> np.ndarray:
        error = np.zeros(3, dtype=float)
        if accel_body_mps2 is not None:
            measured = _unit_or_none(accel_body_mps2)
            if measured is not None:
                expected = navigation_to_body(np.array([0.0, 0.0, 1.0]), self.q)
                error += np.cross(measured, expected)
        if mag_body is not None:
            measured_mag = _unit_or_none(mag_body)
            if measured_mag is not None:
                expected_mag = navigation_to_body(np.array([1.0, 0.0, 0.0]), self.q)
                error += np.cross(measured_mag, expected_mag)
        return error

    def update(self, gyro_body_radps: np.ndarray, accel_body_mps2: np.ndarray | None, dt_s: float, mag_body: np.ndarray | None = None) -> AttitudeEstimate:
        dt = float(dt_s)
        error = self._innovation(accel_body_mps2, mag_body)
        self.integral_error += error * dt
        self.gyro_bias_radps = -self.integral_gain * self.integral_error
        corrected_gyro = np.asarray(gyro_body_radps, dtype=float).reshape(3) - self.gyro_bias_radps + self.proportional_gain * error
        self.predict(corrected_gyro, dt)
        return self.estimate()


class MadgwickAttitudeFilter(ComplementaryAttitudeFilter):
    """Madgwick-style gradient descent IMU attitude filter."""

    def __init__(self, q_nav_from_body: np.ndarray | None = None, beta: float = 0.08):
        super().__init__(q_nav_from_body=q_nav_from_body, accel_gain=0.0, mag_gain=0.0)
        self.beta = float(beta)

    @staticmethod
    def _quat_derivative(q: np.ndarray, gyro_body_radps: np.ndarray) -> np.ndarray:
        omega = np.r_[0.0, np.asarray(gyro_body_radps, dtype=float).reshape(3)]
        return 0.5 * quaternion_multiply(q, omega)

    @staticmethod
    def _accelerometer_gradient(q: np.ndarray, accel_body_mps2: np.ndarray) -> np.ndarray:
        accel = _unit_or_none(accel_body_mps2)
        if accel is None:
            return np.zeros(4, dtype=float)
        qw, qx, qy, qz = normalize_quaternion(q)
        ax, ay, az = accel
        f = np.array(
            [
                2.0 * (qx * qz - qw * qy) - ax,
                2.0 * (qw * qx + qy * qz) - ay,
                2.0 * (0.5 - qx * qx - qy * qy) - az,
            ],
            dtype=float,
        )
        jac = np.array(
            [
                [-2.0 * qy, 2.0 * qz, -2.0 * qw, 2.0 * qx],
                [2.0 * qx, 2.0 * qw, 2.0 * qz, 2.0 * qy],
                [0.0, -4.0 * qx, -4.0 * qy, 0.0],
            ],
            dtype=float,
        )
        gradient = jac.T @ f
        norm = float(np.linalg.norm(gradient))
        return gradient / norm if norm > 1e-12 else gradient

    def update(self, gyro_body_radps: np.ndarray, accel_body_mps2: np.ndarray | None, dt_s: float, mag_body: np.ndarray | None = None) -> AttitudeEstimate:
        del mag_body
        gradient = np.zeros(4, dtype=float) if accel_body_mps2 is None else self._accelerometer_gradient(self.q, accel_body_mps2)
        q_dot = self._quat_derivative(self.q, gyro_body_radps) - self.beta * gradient
        self.q = normalize_quaternion(self.q + q_dot * float(dt_s))
        return self.estimate()
