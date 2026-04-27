from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite


def inflate_covariance(P: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return nearest_positive_definite(P)
    return nearest_positive_definite(P * float(scale))
