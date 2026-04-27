from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite, safe_inverse
from .linear_kf import LinearKalmanFilter


class InformationKalmanFilter(LinearKalmanFilter):
    """Information-form linear filter with covariance synchronization for shared APIs."""

    filter_type = "information"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.Y = safe_inverse(self.P)
        self.y_info = self.Y @ self.x

    def _sync_from_covariance(self) -> None:
        self.P = nearest_positive_definite(self.P)
        self.Y = safe_inverse(self.P)
        self.y_info = self.Y @ self.x

    def _sync_to_covariance(self) -> None:
        self.P = nearest_positive_definite(safe_inverse(self.Y))
        self.x = self.P @ self.y_info

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        self._sync_to_covariance()
        super().predict(imu_accel, dt, dropout)
        self._sync_from_covariance()

    def linear_update(
        self,
        z: np.ndarray,
        H: np.ndarray,
        R: np.ndarray,
        source: str,
        wrap_indices: list[int] | None = None,
        gate: float | None = None,
        extra: dict[str, float] | None = None,
    ) -> dict[str, float]:
        self._sync_to_covariance()
        z = np.asarray(z, dtype=float).reshape(-1)
        H = np.asarray(H, dtype=float)
        R = np.asarray(R, dtype=float)
        predicted = H @ self.x
        residual = z - predicted
        for idx in wrap_indices or []:
            from ..math_utils import wrap_angle

            residual[idx] = float(wrap_angle(residual[idx]))
        S = H @ self.P @ H.T + R
        threshold = float(self.profile.gate if gate is None else gate)
        from ..fusion.gating import gate_measurement

        accepted, nis = gate_measurement(residual, S, threshold)
        if accepted:
            R_inv = safe_inverse(R)
            self.Y = self.Y + H.T @ R_inv @ H
            self.y_info = self.y_info + H.T @ R_inv @ z
            self._sync_to_covariance()
            self._sync_from_covariance()
        return self._innovation_record(source, residual, S, accepted, nis, threshold, extra)
