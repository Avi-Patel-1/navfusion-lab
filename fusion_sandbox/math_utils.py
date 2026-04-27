from __future__ import annotations

import math
from typing import Callable

import numpy as np


def wrap_angle(angle_rad: float | np.ndarray) -> float | np.ndarray:
    """Wrap radians to [-pi, pi)."""
    return (angle_rad + math.pi) % (2.0 * math.pi) - math.pi


def skew_symmetric(v: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(v, dtype=float).reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


def numerical_jacobian(func: Callable[[np.ndarray], np.ndarray], x: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    y0 = np.asarray(func(x), dtype=float)
    jac = np.zeros((y0.size, x.size), dtype=float)
    for i in range(x.size):
        step = np.zeros_like(x)
        step[i] = eps
        yp = np.asarray(func(x + step), dtype=float)
        ym = np.asarray(func(x - step), dtype=float)
        jac[:, i] = ((yp - ym) / (2.0 * eps)).reshape(-1)
    return jac


def symmetrize(matrix: np.ndarray) -> np.ndarray:
    return 0.5 * (matrix + matrix.T)


def nearest_positive_definite(matrix: np.ndarray, floor: float = 1e-9) -> np.ndarray:
    """Symmetrize and floor covariance eigenvalues."""
    P = symmetrize(np.asarray(matrix, dtype=float))
    try:
        vals, vecs = np.linalg.eigh(P)
    except np.linalg.LinAlgError:
        return np.eye(P.shape[0]) * floor
    vals = np.maximum(vals, floor)
    repaired = (vecs * vals) @ vecs.T
    return symmetrize(repaired)


def safe_inverse(matrix: np.ndarray) -> np.ndarray:
    try:
        return np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix)


def vector_rmse(errors: list[np.ndarray] | np.ndarray) -> float:
    arr = np.asarray(errors, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim == 1:
        return float(np.sqrt(np.mean(arr * arr)))
    return float(np.sqrt(np.mean(np.sum(arr * arr, axis=1))))


def interpolate_records(records: list[dict[str, float]], t: float, fields: list[str]) -> np.ndarray:
    """Linear interpolation for monotonically sampled truth records."""
    if not records:
        raise ValueError("cannot interpolate empty record set")
    if t <= float(records[0]["time_s"]):
        return np.array([records[0][field] for field in fields], dtype=float)
    if t >= float(records[-1]["time_s"]):
        return np.array([records[-1][field] for field in fields], dtype=float)
    lo = 0
    hi = len(records) - 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if float(records[mid]["time_s"]) <= t:
            lo = mid
        else:
            hi = mid
    t0 = float(records[lo]["time_s"])
    t1 = float(records[hi]["time_s"])
    alpha = (t - t0) / max(t1 - t0, 1e-12)
    v0 = np.array([records[lo][field] for field in fields], dtype=float)
    v1 = np.array([records[hi][field] for field in fields], dtype=float)
    return (1.0 - alpha) * v0 + alpha * v1
