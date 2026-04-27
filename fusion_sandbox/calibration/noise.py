from __future__ import annotations

from typing import Any

import numpy as np


def _residual_matrix(innovations: list[dict[str, Any]]) -> np.ndarray:
    rows = []
    for row in innovations:
        values = [
            float(row[key])
            for key in sorted(row)
            if key.startswith("residual_") and key.removeprefix("residual_").isdigit() and row[key] not in {"", None}
        ]
        if values:
            rows.append(values)
    if not rows:
        return np.zeros((0, 0), dtype=float)
    width = max(len(row) for row in rows)
    padded = [row + [0.0] * (width - len(row)) for row in rows]
    return np.asarray(padded, dtype=float)


def estimate_noise_from_innovations(innovations: list[dict[str, Any]]) -> dict[str, Any]:
    residuals = _residual_matrix(innovations)
    if residuals.size == 0:
        return {"count": 0, "residual_std": [], "residual_mean": [], "covariance": []}
    return {
        "count": int(residuals.shape[0]),
        "residual_mean": np.mean(residuals, axis=0).tolist(),
        "residual_std": np.std(residuals, axis=0, ddof=1 if residuals.shape[0] > 1 else 0).tolist(),
        "covariance": np.cov(residuals.T).tolist() if residuals.shape[0] > 1 else np.zeros((residuals.shape[1], residuals.shape[1])).tolist(),
    }


def residual_autocorrelation(values: np.ndarray, max_lag: int = 20) -> list[float]:
    x = np.asarray(values, dtype=float).reshape(-1)
    if x.size == 0:
        return []
    x = x - float(np.mean(x))
    denom = float(np.dot(x, x))
    if denom <= 1e-12:
        return [0.0 for _ in range(max_lag + 1)]
    out = []
    for lag in range(max_lag + 1):
        if lag == 0:
            out.append(1.0)
        elif lag < x.size:
            out.append(float(np.dot(x[:-lag], x[lag:]) / denom))
        else:
            out.append(0.0)
    return out


def allan_deviation(samples: np.ndarray, sample_rate_hz: float, cluster_sizes: list[int] | None = None) -> dict[str, list[float]]:
    x = np.asarray(samples, dtype=float).reshape(-1)
    if x.size < 4:
        return {"tau_s": [], "adev": []}
    if cluster_sizes is None:
        max_m = max(2, x.size // 8)
        cluster_sizes = sorted(set(int(v) for v in np.geomspace(1, max_m, num=min(18, max_m))))
    taus: list[float] = []
    adev: list[float] = []
    for m in cluster_sizes:
        if m < 1 or 2 * m >= x.size:
            continue
        n = x.size // m
        trimmed = x[: n * m]
        clusters = trimmed.reshape(n, m).mean(axis=1)
        diffs = np.diff(clusters)
        sigma2 = 0.5 * float(np.mean(diffs * diffs))
        taus.append(float(m / sample_rate_hz))
        adev.append(float(np.sqrt(max(sigma2, 0.0))))
    return {"tau_s": taus, "adev": adev}
