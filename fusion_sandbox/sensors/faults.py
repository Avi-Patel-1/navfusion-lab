from __future__ import annotations

from typing import Any

import numpy as np


def in_windows(t: float, windows: list[list[float]]) -> bool:
    return any(float(start) <= t <= float(end) for start, end in windows)


def outlier_offset(t: float, bursts: list[dict[str, Any]], key: str, dimension: int) -> tuple[np.ndarray, bool]:
    total = np.zeros(dimension, dtype=float)
    active = False
    for burst in bursts:
        start = float(burst.get("start_s", 0.0))
        duration = float(burst.get("duration_s", 0.0))
        if start <= t <= start + duration:
            total += np.array(burst.get(key, [0.0] * dimension), dtype=float)
            active = True
    return total, active
