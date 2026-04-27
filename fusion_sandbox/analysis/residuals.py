from __future__ import annotations


def residual_summary(innovations: list[dict[str, float]]) -> dict[str, float]:
    values = [float(row["residual_norm"]) for row in innovations if "residual_norm" in row]
    if not values:
        return {"count": 0.0, "mean": 0.0, "max": 0.0}
    return {"count": float(len(values)), "mean": sum(values) / len(values), "max": max(values)}
