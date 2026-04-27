from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..math_utils import interpolate_records, vector_rmse


def _window_mask(times: np.ndarray, windows: list[list[float]]) -> np.ndarray:
    mask = np.zeros(times.shape, dtype=bool)
    for start, end in windows:
        mask |= (times >= float(start)) & (times <= float(end))
    return mask


def _recovery_time(times: np.ndarray, errors: np.ndarray, windows: list[list[float]]) -> float:
    if not windows or errors.size == 0:
        return 0.0
    baseline = float(np.median(errors[: max(5, min(30, errors.size))])) + 2.0
    end = max(float(window[1]) for window in windows)
    after = np.where(times >= end)[0]
    for idx in after:
        if errors[idx] <= baseline:
            return float(times[idx] - end)
    return float(times[-1] - end)


def compute_metrics(
    estimates: list[dict[str, float]],
    innovations: list[dict[str, float]],
    truth: list[dict[str, float]],
    measurements: list[dict[str, float]],
    config: dict[str, Any],
) -> dict[str, Any]:
    est_pos = np.array([[row["px"], row["py"], row["pz"]] for row in estimates], dtype=float)
    est_vel = np.array([[row["vx"], row["vy"], row["vz"]] for row in estimates], dtype=float)
    times = np.array([row["time_s"] for row in estimates], dtype=float)
    truth_pos = np.vstack([interpolate_records(truth, float(t), ["px", "py", "pz"]) for t in times])
    truth_vel = np.vstack([interpolate_records(truth, float(t), ["vx", "vy", "vz"]) for t in times])
    pos_errors = est_pos - truth_pos
    vel_errors = est_vel - truth_vel
    pos_norm = np.linalg.norm(pos_errors, axis=1)
    vel_norm = np.linalg.norm(vel_errors, axis=1)
    dropout_windows = config["sensors"].get("gps", {}).get("dropout_windows_s", [])
    dropout_mask = _window_mask(times, dropout_windows)
    nis = np.array([row["nis"] for row in innovations if "nis" in row], dtype=float)
    residuals = np.array([row["residual_norm"] for row in innovations if "residual_norm" in row], dtype=float)
    accepted = int(sum(1 for row in innovations if float(row.get("accepted", 0.0)) >= 0.5))
    rejected = int(sum(1 for row in innovations if float(row.get("accepted", 0.0)) < 0.5))
    invalid_measurements = int(sum(1 for row in measurements if float(row.get("valid", 1.0)) < 0.5))
    covariance_trace = np.array([row["covariance_trace"] for row in estimates], dtype=float)
    nees = np.array([row.get("nees_position", 0.0) for row in estimates], dtype=float)
    bias_errors = np.array([[row.get("bax_error", 0.0), row.get("bay_error", 0.0), row.get("baz_error", 0.0)] for row in estimates], dtype=float)
    metrics = {
        "position_rmse_m": vector_rmse(pos_errors),
        "velocity_rmse_mps": vector_rmse(vel_errors),
        "bias_rmse_mps2": vector_rmse(bias_errors),
        "final_position_error_m": float(pos_norm[-1]),
        "final_velocity_error_mps": float(vel_norm[-1]),
        "max_position_error_m": float(np.max(pos_norm)),
        "max_error_during_dropout_m": float(np.max(pos_norm[dropout_mask])) if np.any(dropout_mask) else 0.0,
        "dropout_recovery_time_s": _recovery_time(times, pos_norm, dropout_windows),
        "nis_mean": float(np.mean(nis)) if nis.size else 0.0,
        "nis_p95": float(np.percentile(nis, 95)) if nis.size else 0.0,
        "mean_nees_position": float(np.mean(nees)) if nees.size else 0.0,
        "accepted_updates": accepted,
        "rejected_updates": rejected,
        "invalid_measurements": invalid_measurements,
        "residual_mean": float(np.mean(residuals)) if residuals.size else 0.0,
        "residual_std": float(np.std(residuals)) if residuals.size else 0.0,
        "mean_covariance_trace": float(np.mean(covariance_trace)) if covariance_trace.size else 0.0,
        "final_covariance_trace": float(covariance_trace[-1]) if covariance_trace.size else 0.0,
        "diverged": bool(np.any(~np.isfinite(pos_norm)) or np.max(pos_norm) > 5000.0 or math.isnan(float(covariance_trace[-1]))),
    }
    by_sensor: dict[str, dict[str, int]] = {}
    for row in innovations:
        source = str(row.get("source", "unknown")).split("_")[0]
        counts = by_sensor.setdefault(source, {"accepted": 0, "rejected": 0})
        if float(row.get("accepted", 0.0)) >= 0.5:
            counts["accepted"] += 1
        else:
            counts["rejected"] += 1
    metrics["updates_by_sensor"] = by_sensor
    return metrics
