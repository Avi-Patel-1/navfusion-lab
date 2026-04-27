from __future__ import annotations

import numpy as np

from ..math_utils import safe_inverse


def normalized_innovation_squared(residual: np.ndarray, covariance: np.ndarray) -> float:
    y = np.asarray(residual, dtype=float).reshape(-1)
    S = np.asarray(covariance, dtype=float)
    return float(y.T @ safe_inverse(S) @ y)


def gate_measurement(residual: np.ndarray, covariance: np.ndarray, gate_threshold: float) -> tuple[bool, float]:
    nis = normalized_innovation_squared(residual, covariance)
    return bool(nis <= float(gate_threshold)), nis
