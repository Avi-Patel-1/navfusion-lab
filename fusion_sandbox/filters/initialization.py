from __future__ import annotations

from typing import Any

import numpy as np


def initial_state_from_truth(truth0: dict[str, float], estimator_config: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    position = np.array([truth0["px"], truth0["py"], truth0["pz"]], dtype=float)
    velocity = np.array([truth0["vx"], truth0["vy"], truth0["vz"]], dtype=float)
    position += np.array(estimator_config.get("initial_position_error_m", [0.0, 0.0, 0.0]), dtype=float)
    velocity += np.array(estimator_config.get("initial_velocity_error_mps", [0.0, 0.0, 0.0]), dtype=float)
    return position, velocity
