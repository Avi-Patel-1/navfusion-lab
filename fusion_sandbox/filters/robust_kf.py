from __future__ import annotations

import numpy as np

from .ekf import ExtendedKalmanFilter


class RobustHuberKalmanFilter(ExtendedKalmanFilter):
    """EKF variant that downweights large residuals using a Huber-style scale."""

    filter_type = "robust"

    def linear_update(self, z, H, R, source, wrap_indices=None, gate=None, extra=None):
        z_arr = np.asarray(z, dtype=float).reshape(-1)
        H_arr = np.asarray(H, dtype=float)
        R_arr = np.asarray(R, dtype=float)
        residual = z_arr - H_arr @ self.x
        for idx in wrap_indices or []:
            from ..math_utils import wrap_angle

            residual[idx] = float(wrap_angle(residual[idx]))
        sigma = np.sqrt(np.maximum(np.diag(R_arr), 1e-12))
        normalized = np.abs(residual) / sigma
        huber_k = 1.8
        inflation = np.maximum(1.0, normalized / huber_k)
        robust_R = R_arr * np.outer(inflation, inflation)
        record = super().linear_update(z_arr, H_arr, robust_R, source, wrap_indices=wrap_indices, gate=gate, extra=extra)
        record["huber_max_inflation"] = float(np.max(inflation))
        return record


class InnovationAdaptiveKalmanFilter(ExtendedKalmanFilter):
    """EKF variant that adapts measurement covariance from recent NIS behavior."""

    filter_type = "adaptive"

    def __init__(self, *args, window: int = 12, **kwargs):
        super().__init__(*args, **kwargs)
        self.window = window
        self._nis_history: list[float] = []
        self.measurement_scale = 1.0

    def linear_update(self, z, H, R, source, wrap_indices=None, gate=None, extra=None):
        scaled_R = np.asarray(R, dtype=float) * self.measurement_scale
        record = super().linear_update(z, H, scaled_R, source, wrap_indices=wrap_indices, gate=gate, extra=extra)
        dim = max(np.asarray(z, dtype=float).size, 1)
        self._nis_history.append(float(record["nis"]) / dim)
        self._nis_history = self._nis_history[-self.window :]
        mean_nis = float(np.mean(self._nis_history))
        if mean_nis > 2.0:
            self.measurement_scale = min(self.measurement_scale * 1.08, 25.0)
        elif mean_nis < 0.5 and len(self._nis_history) >= self.window:
            self.measurement_scale = max(self.measurement_scale * 0.96, 0.25)
        record["adaptive_measurement_scale"] = float(self.measurement_scale)
        return record
