from __future__ import annotations

import numpy as np


def covariance_positive_diagonal(P: np.ndarray) -> bool:
    return bool(np.all(np.diag(P) > 0.0) and np.allclose(P, P.T, atol=1e-8))
