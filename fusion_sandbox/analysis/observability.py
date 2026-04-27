from __future__ import annotations


def low_maneuver_fraction(truth: list[dict[str, float]], threshold_mps2: float = 0.12) -> float:
    if not truth:
        return 0.0
    low = 0
    for row in truth:
        accel_norm = (row["ax"] ** 2 + row["ay"] ** 2 + row["az"] ** 2) ** 0.5
        low += int(accel_norm < threshold_mps2)
    return low / len(truth)
