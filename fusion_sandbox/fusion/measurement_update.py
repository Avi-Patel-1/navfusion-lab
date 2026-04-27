from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite, safe_inverse


def joseph_update(x: np.ndarray, P: np.ndarray, H: np.ndarray, R: np.ndarray, residual: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    S = H @ P @ H.T + R
    K = P @ H.T @ safe_inverse(S)
    updated_x = x + K @ residual
    I = np.eye(P.shape[0])
    updated_P = (I - K @ H) @ P @ (I - K @ H).T + K @ R @ K.T
    return updated_x, nearest_positive_definite(updated_P), K
