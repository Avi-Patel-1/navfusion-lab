from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from .schema import ALLOWED_FILTERS, ALLOWED_PROFILES, ALLOWED_TRAJECTORIES


DEFAULT_CONFIG: dict[str, Any] = {
    "name": "baseline_ekf_imu_gps_baro",
    "seed": 314,
    "duration_s": 60.0,
    "truth_sample_rate_hz": 20.0,
    "trajectory": {
        "profile": "mixed",
        "mode": "3d",
        "initial_position_m": [0.0, 0.0, 120.0],
        "initial_velocity_mps": [18.0, 1.5, 0.2],
        "figure_eight_radius_m": 90.0,
    },
    "sensors": {
        "imu": {
            "enabled": True,
            "rate_hz": 20.0,
            "accel_noise_std_mps2": 0.045,
            "gyro_noise_std_radps": 0.002,
            "accel_bias_mps2": [0.035, -0.025, 0.02],
            "gyro_bias_radps": [0.0, 0.0, 0.001],
            "accel_bias_walk_std_mps2_sqrt_s": 0.0007,
            "scale_factor": [1.0, 1.0, 1.0],
        },
        "gps": {
            "enabled": True,
            "rate_hz": 1.0,
            "position_noise_std_m": 1.8,
            "velocity_noise_std_mps": 0.25,
            "latency_s": 0.0,
            "dropout_windows_s": [[28.0, 36.0]],
            "outlier_bursts": [{"start_s": 43.0, "duration_s": 0.15, "position_offset_m": [45.0, -30.0, 20.0]}],
        },
        "barometer": {
            "enabled": True,
            "rate_hz": 5.0,
            "altitude_noise_std_m": 0.7,
            "bias_m": 0.0,
            "bias_walk_std_m_sqrt_s": 0.004,
        },
        "magnetometer": {
            "enabled": False,
            "rate_hz": 2.0,
            "heading_noise_std_rad": 0.035,
            "disturbances": [],
        },
        "range_beacon": {
            "enabled": False,
            "rate_hz": 2.0,
            "range_noise_std_m": 0.9,
            "beacons_m": [[0.0, 0.0, 100.0], [250.0, -120.0, 125.0], [150.0, 220.0, 95.0]],
        },
        "wheel_odometry": {
            "enabled": False,
            "rate_hz": 10.0,
            "velocity_noise_std_mps": 0.2,
        },
        "radar_altimeter": {
            "enabled": False,
            "rate_hz": 8.0,
            "altitude_noise_std_m": 0.35,
            "terrain_altitude_m": 0.0,
            "bias_m": 0.0,
            "warmup_s": 0.0,
            "packet_loss_probability": 0.0,
            "timestamp_jitter_std_s": 0.0,
            "dropout_windows_s": [],
            "outlier_bursts": [],
        },
        "doppler_velocity": {
            "enabled": False,
            "rate_hz": 5.0,
            "velocity_noise_std_mps": 0.12,
            "scale_factor": [1.0, 1.0, 1.0],
            "axis_misalignment_rad": [0.0, 0.0, 0.0],
            "warmup_s": 0.0,
            "packet_loss_probability": 0.0,
            "timestamp_jitter_std_s": 0.0,
            "dropout_windows_s": [],
            "outlier_bursts": [],
        },
        "gnss_pseudorange": {
            "enabled": False,
            "rate_hz": 1.0,
            "pseudorange_noise_std_m": 1.2,
            "satellites_m": [
                [15600000.0, 7540000.0, 20140000.0],
                [-18760000.0, 13400000.0, 13020000.0],
                [17610000.0, -14630000.0, 13480000.0],
                [-19170000.0, -6100000.0, 18390000.0],
            ],
            "receiver_clock_bias_m": 0.0,
            "clock_bias_correction_m": 0.0,
            "clock_drift_mps": 0.0,
            "packet_loss_probability": 0.0,
            "timestamp_jitter_std_s": 0.0,
            "dropout_windows_s": [],
            "outlier_bursts": [],
        },
    },
    "estimator": {
        "update_rate_hz": 20.0,
        "filters": ["ekf"],
        "profiles": ["aggressive", "nominal", "conservative", "dropout_robust", "high_bias"],
        "initial_position_error_m": [4.0, -3.0, 1.5],
        "initial_velocity_error_mps": [0.5, -0.25, 0.15],
        "max_stale_s": 1.5,
    },
    "experiments": [],
}


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _legacy_to_nested(config: dict[str, Any]) -> dict[str, Any]:
    cfg = copy.deepcopy(config)
    trajectory = cfg.setdefault("trajectory", {})
    sensors = cfg.setdefault("sensors", {})
    estimator = cfg.setdefault("estimator", {})
    if "dt_s" in cfg:
        cfg["truth_sample_rate_hz"] = 1.0 / float(cfg.pop("dt_s"))
        estimator.setdefault("update_rate_hz", cfg["truth_sample_rate_hz"])
    for old, new in [("initial_position_m", "initial_position_m"), ("initial_velocity_mps", "initial_velocity_mps")]:
        if old in cfg:
            trajectory[new] = cfg.pop(old)
    imu = sensors.setdefault("imu", {})
    if "accel_bias_mps2" in cfg:
        imu["accel_bias_mps2"] = cfg.pop("accel_bias_mps2")
    if "imu_noise_std_mps2" in cfg:
        imu["accel_noise_std_mps2"] = cfg.pop("imu_noise_std_mps2")
    gps = sensors.setdefault("gps", {})
    if "gps_noise_std_m" in cfg:
        gps["position_noise_std_m"] = cfg.pop("gps_noise_std_m")
    if "gps_period_s" in cfg:
        gps["rate_hz"] = 1.0 / float(cfg.pop("gps_period_s"))
    if "gps_dropout_window_s" in cfg:
        gps["dropout_windows_s"] = [cfg.pop("gps_dropout_window_s")]
    outlier_time = cfg.pop("gps_outlier_time_s", None)
    outlier = cfg.pop("gps_outlier_m", None)
    if outlier_time is not None and outlier is not None:
        gps["outlier_bursts"] = [{"start_s": outlier_time, "duration_s": 0.15, "position_offset_m": outlier}]
    baro = sensors.setdefault("barometer", {})
    if "baro_noise_std_m" in cfg:
        baro["altitude_noise_std_m"] = cfg.pop("baro_noise_std_m")
    if "baro_period_s" in cfg:
        baro["rate_hz"] = 1.0 / float(cfg.pop("baro_period_s"))
    return cfg


def normalize_config(config: dict[str, Any], overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    converted = _legacy_to_nested(config)
    merged = deep_merge(DEFAULT_CONFIG, converted)
    if overrides:
        merged = deep_merge(merged, overrides)
    return merged


def _is_vec(value: Any, length: int) -> bool:
    return isinstance(value, list) and len(value) == length and all(isinstance(v, (int, float)) for v in value)


def _is_nonnegative_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) >= 0.0


def _is_positive_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and float(value) > 0.0


def _validate_nonnegative(config: dict[str, Any], keys: list[str], path: str, errors: list[str]) -> None:
    for key in keys:
        if key in config and not _is_nonnegative_number(config[key]):
            errors.append(f"{path}.{key} must be nonnegative")


def _validate_windows(windows: Any, path: str, errors: list[str]) -> None:
    if not isinstance(windows, list):
        errors.append(f"{path} must be a list")
        return
    for i, window in enumerate(windows):
        if not _is_vec(window, 2) or float(window[1]) < float(window[0]):
            errors.append(f"{path}[{i}] must be [start_s, end_s] with end >= start")


def validate_config(config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    if not _is_positive_number(config.get("duration_s")):
        errors.append("duration_s must be positive")
    if not _is_positive_number(config.get("truth_sample_rate_hz")):
        errors.append("truth_sample_rate_hz must be positive")
    traj = config.get("trajectory", {})
    if traj.get("profile") not in ALLOWED_TRAJECTORIES:
        errors.append(f"trajectory.profile must be one of {sorted(ALLOWED_TRAJECTORIES)}")
    if traj.get("mode") not in {"2d", "3d"}:
        errors.append("trajectory.mode must be 2d or 3d")
    if not _is_vec(traj.get("initial_position_m"), 3):
        errors.append("trajectory.initial_position_m must be length-3 numeric list")
    if not _is_vec(traj.get("initial_velocity_mps"), 3):
        errors.append("trajectory.initial_velocity_mps must be length-3 numeric list")
    sensors = config.get("sensors", {})
    for sensor_name, sensor_cfg in sensors.items():
        if not isinstance(sensor_cfg.get("enabled", False), bool):
            errors.append(f"sensors.{sensor_name}.enabled must be boolean")
        if sensor_cfg.get("enabled", False) and not _is_positive_number(sensor_cfg.get("rate_hz")):
            errors.append(f"sensors.{sensor_name}.rate_hz must be positive")
    imu = sensors.get("imu", {})
    if not _is_vec(imu.get("accel_bias_mps2"), 3):
        errors.append("sensors.imu.accel_bias_mps2 must be length-3 numeric list")
    if not _is_vec(imu.get("scale_factor"), 3):
        errors.append("sensors.imu.scale_factor must be length-3 numeric list")
    if not _is_vec(imu.get("gyro_bias_radps"), 3):
        errors.append("sensors.imu.gyro_bias_radps must be length-3 numeric list")
    _validate_nonnegative(
        imu,
        ["accel_noise_std_mps2", "gyro_noise_std_radps", "accel_bias_walk_std_mps2_sqrt_s"],
        "sensors.imu",
        errors,
    )
    gps = sensors.get("gps", {})
    _validate_nonnegative(gps, ["position_noise_std_m", "velocity_noise_std_mps", "latency_s"], "sensors.gps", errors)
    _validate_windows(gps.get("dropout_windows_s", []), "sensors.gps.dropout_windows_s", errors)
    for i, burst in enumerate(gps.get("outlier_bursts", [])):
        if not _is_nonnegative_number(burst.get("duration_s", 0.0)):
            errors.append(f"sensors.gps.outlier_bursts[{i}].duration_s must be nonnegative")
        if not _is_vec(burst.get("position_offset_m"), 3):
            errors.append(f"sensors.gps.outlier_bursts[{i}].position_offset_m must be length-3 numeric list")
    baro = sensors.get("barometer", {})
    _validate_nonnegative(baro, ["altitude_noise_std_m", "bias_walk_std_m_sqrt_s"], "sensors.barometer", errors)
    mag = sensors.get("magnetometer", {})
    _validate_nonnegative(mag, ["heading_noise_std_rad"], "sensors.magnetometer", errors)
    for i, disturbance in enumerate(mag.get("disturbances", [])):
        if not _is_nonnegative_number(disturbance.get("duration_s", 0.0)):
            errors.append(f"sensors.magnetometer.disturbances[{i}].duration_s must be nonnegative")
    range_cfg = sensors.get("range_beacon", {})
    _validate_nonnegative(range_cfg, ["range_noise_std_m"], "sensors.range_beacon", errors)
    if range_cfg.get("enabled", False) and not range_cfg.get("beacons_m"):
        errors.append("sensors.range_beacon.beacons_m must contain at least one beacon when enabled")
    for i, beacon in enumerate(range_cfg.get("beacons_m", [])):
        if not _is_vec(beacon, 3):
            errors.append(f"sensors.range_beacon.beacons_m[{i}] must be length-3 numeric list")
    wheel = sensors.get("wheel_odometry", {})
    _validate_nonnegative(wheel, ["velocity_noise_std_mps"], "sensors.wheel_odometry", errors)
    radar = sensors.get("radar_altimeter", {})
    _validate_nonnegative(
        radar,
        ["altitude_noise_std_m", "warmup_s", "packet_loss_probability", "timestamp_jitter_std_s"],
        "sensors.radar_altimeter",
        errors,
    )
    if float(radar.get("packet_loss_probability", 0.0)) > 1.0:
        errors.append("sensors.radar_altimeter.packet_loss_probability must be <= 1")
    _validate_windows(radar.get("dropout_windows_s", []), "sensors.radar_altimeter.dropout_windows_s", errors)
    for i, burst in enumerate(radar.get("outlier_bursts", [])):
        if not _is_nonnegative_number(burst.get("duration_s", 0.0)):
            errors.append(f"sensors.radar_altimeter.outlier_bursts[{i}].duration_s must be nonnegative")
        if not isinstance(burst.get("altitude_offset_m", 0.0), (int, float)):
            errors.append(f"sensors.radar_altimeter.outlier_bursts[{i}].altitude_offset_m must be numeric")
    doppler = sensors.get("doppler_velocity", {})
    _validate_nonnegative(
        doppler,
        ["velocity_noise_std_mps", "warmup_s", "packet_loss_probability", "timestamp_jitter_std_s"],
        "sensors.doppler_velocity",
        errors,
    )
    if float(doppler.get("packet_loss_probability", 0.0)) > 1.0:
        errors.append("sensors.doppler_velocity.packet_loss_probability must be <= 1")
    if not _is_vec(doppler.get("scale_factor"), 3):
        errors.append("sensors.doppler_velocity.scale_factor must be length-3 numeric list")
    if not _is_vec(doppler.get("axis_misalignment_rad"), 3):
        errors.append("sensors.doppler_velocity.axis_misalignment_rad must be length-3 numeric list")
    _validate_windows(doppler.get("dropout_windows_s", []), "sensors.doppler_velocity.dropout_windows_s", errors)
    for i, burst in enumerate(doppler.get("outlier_bursts", [])):
        if not _is_nonnegative_number(burst.get("duration_s", 0.0)):
            errors.append(f"sensors.doppler_velocity.outlier_bursts[{i}].duration_s must be nonnegative")
        if not _is_vec(burst.get("velocity_offset_mps", [0.0, 0.0, 0.0]), 3):
            errors.append(f"sensors.doppler_velocity.outlier_bursts[{i}].velocity_offset_mps must be length-3 numeric list")
    gnss = sensors.get("gnss_pseudorange", {})
    _validate_nonnegative(
        gnss,
        ["pseudorange_noise_std_m", "packet_loss_probability", "timestamp_jitter_std_s"],
        "sensors.gnss_pseudorange",
        errors,
    )
    if float(gnss.get("packet_loss_probability", 0.0)) > 1.0:
        errors.append("sensors.gnss_pseudorange.packet_loss_probability must be <= 1")
    if gnss.get("enabled", False) and len(gnss.get("satellites_m", [])) < 4:
        errors.append("sensors.gnss_pseudorange.satellites_m must contain at least four satellites when enabled")
    for i, satellite in enumerate(gnss.get("satellites_m", [])):
        if not _is_vec(satellite, 3):
            errors.append(f"sensors.gnss_pseudorange.satellites_m[{i}] must be length-3 numeric list")
    _validate_windows(gnss.get("dropout_windows_s", []), "sensors.gnss_pseudorange.dropout_windows_s", errors)
    for i, burst in enumerate(gnss.get("outlier_bursts", [])):
        if not _is_nonnegative_number(burst.get("duration_s", 0.0)):
            errors.append(f"sensors.gnss_pseudorange.outlier_bursts[{i}].duration_s must be nonnegative")
        if not isinstance(burst.get("pseudorange_offset_m", 0.0), (int, float)):
            errors.append(f"sensors.gnss_pseudorange.outlier_bursts[{i}].pseudorange_offset_m must be numeric")
    estimator = config.get("estimator", {})
    filters = estimator.get("filters", [])
    profiles = estimator.get("profiles", [])
    if not filters or any(f not in ALLOWED_FILTERS for f in filters):
        errors.append(f"estimator.filters must contain values from {sorted(ALLOWED_FILTERS)}")
    if not profiles or any(p not in ALLOWED_PROFILES for p in profiles):
        errors.append(f"estimator.profiles must contain values from {sorted(ALLOWED_PROFILES)}")
    if not _is_positive_number(estimator.get("update_rate_hz")):
        errors.append("estimator.update_rate_hz must be positive")
    if not _is_nonnegative_number(estimator.get("max_stale_s", 0.0)):
        errors.append("estimator.max_stale_s must be nonnegative")
    if not _is_vec(estimator.get("initial_position_error_m"), 3):
        errors.append("estimator.initial_position_error_m must be length-3 numeric list")
    if not _is_vec(estimator.get("initial_velocity_error_mps"), 3):
        errors.append("estimator.initial_velocity_error_mps must be length-3 numeric list")
    experiments = config.get("experiments", [])
    if not isinstance(experiments, list):
        errors.append("experiments must be a list")
    else:
        base = copy.deepcopy(config)
        base["experiments"] = []
        for i, experiment in enumerate(experiments):
            if not isinstance(experiment, dict):
                errors.append(f"experiments[{i}] must be an object")
                continue
            if not isinstance(experiment.get("name", ""), str) or not experiment.get("name"):
                errors.append(f"experiments[{i}].name must be a non-empty string")
            overrides = experiment.get("overrides", {})
            if not isinstance(overrides, dict):
                errors.append(f"experiments[{i}].overrides must be an object")
                continue
            candidate = deep_merge(base, overrides)
            candidate["experiments"] = []
            for child_error in validate_config(candidate):
                errors.append(f"experiments[{i}].overrides: {child_error}")
    return errors


def load_config(path: str | Path, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    raw = json.loads(Path(path).read_text())
    config = normalize_config(raw, overrides)
    errors = validate_config(config)
    if errors:
        message = "\n".join(f"- {err}" for err in errors)
        raise ValueError(f"Invalid config {path}:\n{message}")
    return config


def validate_config_file(path: str | Path) -> tuple[dict[str, Any], list[str]]:
    raw = json.loads(Path(path).read_text())
    config = normalize_config(raw)
    return config, validate_config(config)
