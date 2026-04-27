from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite
from .linear_kf import LinearKalmanFilter


class SquareRootKalmanFilter(LinearKalmanFilter):
    """Linear Kalman filter that stores and refreshes a Cholesky covariance factor."""

    filter_type = "sqrt"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.S_chol = np.linalg.cholesky(nearest_positive_definite(self.P))

    def _refresh_factor(self) -> None:
        self.P = nearest_positive_definite(self.P)
        self.S_chol = np.linalg.cholesky(self.P)

    def predict(self, imu_accel: np.ndarray, dt: float, dropout: bool = False) -> None:
        super().predict(imu_accel, dt, dropout)
        self._refresh_factor()

    def linear_update(self, *args, **kwargs) -> dict[str, float]:
        record = super().linear_update(*args, **kwargs)
        self._refresh_factor()
        record["sqrt_condition"] = float(np.linalg.cond(self.S_chol))
        return record
