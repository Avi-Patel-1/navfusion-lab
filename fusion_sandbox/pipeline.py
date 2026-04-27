from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np

from .analysis.metrics import compute_metrics
from .calibration.noise import allan_deviation, estimate_noise_from_innovations, residual_autocorrelation
from .config import deep_merge, load_config, normalize_config, validate_config, validate_config_file
from .datasets.c_header import export_run_c_header
from .datasets.events import read_event_csv, read_jsonl, summarize_events, validate_events, write_jsonl
from .fdir.health import analyze_faults
from .experiments.manager import plan_experiment
from .filters import PROFILE_MAP, create_filter
from .filters.initialization import initial_state_from_truth
from .math_utils import wrap_angle
from .model import simulate_navigation
from .reports.csv import write_csv
from .reports.html import write_report
from .reports.json import read_json, write_json
from .reports.svg import write_bar_svg, write_multi_series_svg, write_series_svg, write_xy_svg
from .sensors.faults import in_windows
from .sensors.timing import sample_times
from .trajectory import truth_at


def _event_extra(event: dict[str, float], current_time_s: float) -> dict[str, float]:
    return {
        "time_s": float(current_time_s),
        "measurement_time_s": float(event.get("measurement_time_s", current_time_s)),
        "stale_s": float(current_time_s - float(event.get("measurement_time_s", current_time_s))),
        "sensor_is_outlier": float(event.get("is_outlier", 0.0)),
        "sensor_is_dropout": float(event.get("is_dropout", 0.0)),
    }


def _rejected_record(event: dict[str, float], current_time_s: float, reason: str) -> dict[str, float]:
    return {
        "time_s": float(current_time_s),
        "measurement_time_s": float(event.get("measurement_time_s", current_time_s)),
        "source": f"{event.get('sensor', 'measurement')}_{reason}",
        "nis": 0.0,
        "gate": 0.0,
        "accepted": 0.0,
        "residual_norm": 0.0,
        "innovation_std": 0.0,
        "stale_s": float(current_time_s - float(event.get("measurement_time_s", current_time_s))),
        "sensor_is_outlier": float(event.get("is_outlier", 0.0)),
        "sensor_is_dropout": float(event.get("is_dropout", 0.0)),
    }


def _apply_measurement(filt: Any, event: dict[str, float], current_time_s: float) -> dict[str, float] | None:
    sensor = str(event["sensor"])
    extra = _event_extra(event, current_time_s)
    if sensor == "gps":
        return filt.update_gps(
            np.array([event["x"], event["y"], event["z"]], dtype=float),
            np.array([event["vx"], event["vy"], event["vz"]], dtype=float),
            extra=extra,
        )
    if sensor == "barometer":
        return filt.update_baro(float(event["altitude_m"]), extra=extra)
    if sensor == "range_beacon" and hasattr(filt, "update_range_beacon"):
        beacon = np.array([event["beacon_x"], event["beacon_y"], event["beacon_z"]], dtype=float)
        return filt.update_range_beacon(float(event["range_m"]), beacon, extra=extra)
    if sensor == "magnetometer" and hasattr(filt, "update_magnetometer"):
        return filt.update_magnetometer(float(event["heading_rad"]), extra=extra)
    if sensor == "wheel_odometry" and hasattr(filt, "update_wheel"):
        return filt.update_wheel(np.array([event["vx"], event["vy"]], dtype=float), extra=extra)
    if sensor == "radar_altimeter" and hasattr(filt, "update_radar_altimeter"):
        return filt.update_radar_altimeter(float(event["altitude_agl_m"]), float(event.get("terrain_altitude_m", 0.0)), extra=extra)
    if sensor == "doppler_velocity" and hasattr(filt, "update_doppler_velocity"):
        return filt.update_doppler_velocity(np.array([event["vx"], event["vy"], event["vz"]], dtype=float), extra=extra)
    if sensor == "gnss_pseudorange" and hasattr(filt, "update_gnss_pseudorange"):
        satellite = np.array([event["satellite_x"], event["satellite_y"], event["satellite_z"]], dtype=float)
        return filt.update_gnss_pseudorange(float(event["pseudorange_m"]), satellite, float(event.get("clock_bias_correction_m", 0.0)), extra=extra)
    return None


def _estimate_row(filt: Any, tr: dict[str, float], latest_imu: dict[str, float] | None, filter_type: str, profile_name: str) -> dict[str, float]:
    truth_pos = np.array([tr["px"], tr["py"], tr["pz"]], dtype=float)
    truth_vel = np.array([tr["vx"], tr["vy"], tr["vz"]], dtype=float)
    pos_error = filt.x[:3] - truth_pos
    vel_error = filt.x[3:6] - truth_vel
    true_bias = np.array(
        [
            0.0 if latest_imu is None else latest_imu.get("bax_true", 0.0),
            0.0 if latest_imu is None else latest_imu.get("bay_true", 0.0),
            0.0 if latest_imu is None else latest_imu.get("baz_true", 0.0),
        ],
        dtype=float,
    )
    bias_error = filt.x[6:9] - true_bias
    return {
        "time_s": float(tr["time_s"]),
        "filter": filter_type,
        "profile": profile_name,
        "px": float(filt.x[0]),
        "py": float(filt.x[1]),
        "pz": float(filt.x[2]),
        "vx": float(filt.x[3]),
        "vy": float(filt.x[4]),
        "vz": float(filt.x[5]),
        "bax": float(filt.x[6]),
        "bay": float(filt.x[7]),
        "baz": float(filt.x[8]),
        "yaw_est_rad": float(filt.yaw_estimate),
        "yaw_error_rad": float(wrap_angle(filt.yaw_estimate - float(tr.get("yaw_rad", 0.0)))),
        "pos_error_m": float(np.linalg.norm(pos_error)),
        "velocity_error_mps": float(np.linalg.norm(vel_error)),
        "bax_error": float(bias_error[0]),
        "bay_error": float(bias_error[1]),
        "baz_error": float(bias_error[2]),
        "sigma_pos_m": filt.position_sigma(),
        "sigma_velocity_mps": filt.velocity_sigma(),
        "covariance_trace": filt.covariance_trace(),
        "nees_position": filt.nees_proxy(pos_error),
    }


def _write_filter_plots(out: Path, suffix: str, truth: list[dict[str, float]], estimates: list[dict[str, float]], innovations: list[dict[str, float]], config: dict[str, Any]) -> None:
    times = [row["time_s"] for row in estimates]
    dropout_windows = config["sensors"].get("gps", {}).get("dropout_windows_s", [])
    write_series_svg(out / "plots" / f"position_error_{suffix}.svg", times, [row["pos_error_m"] for row in estimates], f"{suffix} position error", "m", dropout_windows)
    write_series_svg(out / "plots" / f"velocity_error_{suffix}.svg", times, [row["velocity_error_mps"] for row in estimates], f"{suffix} velocity error", "m/s", dropout_windows)
    write_multi_series_svg(
        out / "plots" / f"altitude_{suffix}.svg",
        times,
        [("truth", [row["pz"] for row in truth[: len(estimates)]]), ("estimate", [row["pz"] for row in estimates])],
        f"{suffix} altitude",
        "m",
        dropout_windows,
    )
    write_multi_series_svg(
        out / "plots" / f"covariance_bounds_{suffix}.svg",
        times,
        [("position error", [row["pos_error_m"] for row in estimates]), ("sqrt trace Ppos", [row["sigma_pos_m"] for row in estimates])],
        f"{suffix} covariance bounds",
        "m",
        dropout_windows,
    )
    write_series_svg(out / "plots" / f"nees_{suffix}.svg", times, [row["nees_position"] for row in estimates], f"{suffix} NEES proxy", "value", dropout_windows)
    if innovations:
        it = [row["time_s"] for row in innovations]
        write_series_svg(out / "plots" / f"nis_{suffix}.svg", it, [row["nis"] for row in innovations], f"{suffix} NIS", "value")
        write_series_svg(out / "plots" / f"residuals_{suffix}.svg", it, [row["residual_norm"] for row in innovations], f"{suffix} residuals", "norm")
    write_multi_series_svg(
        out / "plots" / f"bias_estimates_{suffix}.svg",
        times,
        [("bax", [row["bax"] for row in estimates]), ("bay", [row["bay"] for row in estimates]), ("baz", [row["baz"] for row in estimates])],
        f"{suffix} accelerometer bias estimates",
        "m/s^2",
    )
    if suffix.startswith("ekf_"):
        legacy = suffix.removeprefix("ekf_")
        write_series_svg(out / "plots" / f"{legacy}_position_error.svg", times, [row["pos_error_m"] for row in estimates], f"{legacy} position error", "m", dropout_windows)
        write_series_svg(out / "plots" / f"{legacy}_sigma_position.svg", times, [row["sigma_pos_m"] for row in estimates], f"{legacy} covariance trace", "sqrt trace P")


def _run_filter_profile(
    filter_type: str,
    profile_name: str,
    truth: list[dict[str, float]],
    measurements: list[dict[str, float]],
    config: dict[str, Any],
    out: Path,
) -> dict[str, Any]:
    estimator_cfg = config["estimator"]
    initial_position, initial_velocity = initial_state_from_truth(truth[0], estimator_cfg)
    filt = create_filter(filter_type, profile_name, initial_position, initial_velocity)
    suffix = f"{filter_type}_{profile_name}"
    max_stale = float(estimator_cfg.get("max_stale_s", 1.5))
    dropout_windows = config["sensors"].get("gps", {}).get("dropout_windows_s", [])
    event_index = 0
    latest_imu: dict[str, float] | None = None
    ready_measurements: list[dict[str, float]] = []
    estimates: list[dict[str, float]] = []
    innovations: list[dict[str, float]] = []
    last_t = float(truth[0]["time_s"])
    for tr in truth:
        t = float(tr["time_s"])
        while event_index < len(measurements) and float(measurements[event_index]["time_s"]) <= t + 1e-10:
            event = measurements[event_index]
            if event["sensor"] == "imu":
                latest_imu = event
            else:
                ready_measurements.append(event)
            event_index += 1
        if latest_imu is None:
            imu_accel = np.array([tr["ax"], tr["ay"], tr["az"]], dtype=float)
        else:
            imu_accel = np.array([latest_imu["ax"], latest_imu["ay"], latest_imu["az"]], dtype=float)
        dt = t - last_t
        filt.predict(imu_accel, dt, dropout=in_windows(t, dropout_windows))
        last_t = t
        still_waiting: list[dict[str, float]] = []
        for event in ready_measurements:
            if float(event.get("valid", 1.0)) < 0.5:
                continue
            stale = t - float(event.get("measurement_time_s", t))
            if stale > max_stale:
                innovations.append(_rejected_record(event, t, "stale"))
                continue
            rec = _apply_measurement(filt, event, t)
            if rec is None:
                innovations.append(_rejected_record(event, t, "unsupported"))
            else:
                innovations.append(rec)
        ready_measurements = still_waiting
        estimates.append(_estimate_row(filt, tr, latest_imu, filter_type, profile_name))
    write_csv(out / f"estimates_{suffix}.csv", estimates)
    write_csv(out / f"innovations_{suffix}.csv", innovations)
    metrics = compute_metrics(estimates, innovations, truth, measurements, config)
    metrics.update({"filter": filter_type, "profile": profile_name, "mode": PROFILE_MAP[profile_name].mode})
    write_json(out / f"metrics_{suffix}.json", metrics)
    _write_filter_plots(out, suffix, truth, estimates, innovations, config)
    return metrics


def _estimator_truth(config: dict[str, Any], truth: list[dict[str, float]]) -> list[dict[str, float]]:
    rate = float(config["estimator"].get("update_rate_hz", config["truth_sample_rate_hz"]))
    return [truth_at(truth, t) for t in sample_times(float(truth[-1]["time_s"]), rate, include_endpoint=True)]


def _write_run_level_plots(out: Path, truth: list[dict[str, float]], measurements: list[dict[str, float]], results: list[dict[str, Any]]) -> None:
    write_xy_svg(out / "plots" / "trajectory_xy.svg", [row["px"] for row in truth], [row["py"] for row in truth], "Truth trajectory")
    write_series_svg(out / "plots" / "altitude_truth.svg", [row["time_s"] for row in truth], [row["pz"] for row in truth], "Truth altitude", "m")
    labels = [f"{row['filter']}_{row['profile']}" for row in results]
    rmse = [float(row["position_rmse_m"]) for row in results]
    write_bar_svg(out / "plots" / "filter_comparison_bar.svg", labels, rmse, "Filter comparison", "position RMSE (m)")
    accepted = sum(int(row.get("accepted_updates", 0)) for row in results)
    rejected = sum(int(row.get("rejected_updates", 0)) for row in results)
    invalid = sum(1 for row in measurements if float(row.get("valid", 1.0)) < 0.5)
    write_bar_svg(out / "plots" / "accepted_rejected_measurements.svg", ["accepted", "rejected", "invalid"], [accepted, rejected, invalid], "Measurement decisions", "count")
    by_profile: dict[str, list[float]] = {}
    for row in results:
        by_profile.setdefault(str(row["profile"]), []).append(float(row["position_rmse_m"]))
    write_bar_svg(
        out / "plots" / "tuning_profile_comparison.svg",
        list(by_profile.keys()),
        [float(np.mean(values)) for values in by_profile.values()],
        "Tuning profile comparison",
        "mean position RMSE (m)",
    )


def _write_manifest(out: Path, summary: dict[str, Any]) -> None:
    files = sorted(str(path.relative_to(out)) for path in out.rglob("*") if path.is_file() and path.name != "manifest.json")
    manifest = {
        "name": summary.get("name", ""),
        "best_result": summary.get("best_result", summary.get("best_profile", "")),
        "result_count": len(summary.get("results", [])),
        "file_count": len(files) + 1,
        "files": files + ["manifest.json"],
    }
    write_json(out / "manifest.json", manifest)


def run_config(config: dict[str, Any], out_dir: str | Path) -> dict[str, Any]:
    config = normalize_config(config)
    errors = validate_config(config)
    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(f"- {err}" for err in errors))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth, measurements = simulate_navigation(config)
    estimator_truth = _estimator_truth(config, truth)
    write_csv(out / "truth.csv", truth)
    write_csv(out / "measurements.csv", measurements)
    write_json(out / "config_resolved.json", config)
    results: list[dict[str, Any]] = []
    for filter_type in config["estimator"]["filters"]:
        for profile_name in config["estimator"]["profiles"]:
            results.append(_run_filter_profile(filter_type, profile_name, estimator_truth, measurements, config, out))
    best = min(results, key=lambda row: float(row["position_rmse_m"]))
    summary = {
        "name": config.get("name", "run"),
        "samples": len(truth),
        "measurement_count": len(measurements),
        "filters": config["estimator"]["filters"],
        "profiles_requested": config["estimator"]["profiles"],
        "results": results,
        "profiles": results,
        "best_result": f"{best['filter']}_{best['profile']}",
        "best_profile": best["profile"],
    }
    _write_run_level_plots(out, truth, measurements, results)
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    _write_manifest(out, summary)
    return summary


def run_experiment(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    return run_config(config, out_dir)


def compare_experiment(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    base = load_config(config_path)
    experiments = base.get("experiments") or [{"name": base.get("name", "comparison"), "overrides": {}}]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    aggregate: list[dict[str, Any]] = []
    for idx, experiment in enumerate(experiments):
        name = str(experiment.get("name", f"experiment_{idx+1}"))
        overrides = copy.deepcopy(experiment.get("overrides", {}))
        cfg = deep_merge(base, overrides)
        cfg["name"] = name
        cfg["experiments"] = []
        summary = run_config(cfg, out / name)
        for result in summary["results"]:
            aggregate.append({"experiment": name, **result})
    best = min(aggregate, key=lambda row: float(row["position_rmse_m"]))
    labels = [f"{row['experiment']}_{row['filter']}_{row['profile']}" for row in aggregate]
    write_bar_svg(out / "plots" / "comparison_position_rmse.svg", labels, [row["position_rmse_m"] for row in aggregate], "Comparison position RMSE", "m")
    summary = {"name": "comparison", "results": aggregate, "best_result": f"{best['experiment']}_{best['filter']}_{best['profile']}"}
    write_json(out / "summary.json", summary)
    write_report(out, summary)
    _write_manifest(out, summary)
    return summary


def sweep_tuning(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    config["estimator"]["filters"] = ["ekf"]
    config["estimator"]["profiles"] = list(PROFILE_MAP.keys())
    config["name"] = f"{config.get('name', 'run')}_tuning_sweep"
    return run_config(config, out_dir)


def summarize_run(summary_path: str | Path) -> dict[str, Any]:
    summary = read_json(Path(summary_path))
    rows = summary.get("results", [])
    compact = {
        "name": summary.get("name", ""),
        "best_result": summary.get("best_result", ""),
        "result_count": len(rows),
        "position_rmse_m": {f"{row.get('filter')}_{row.get('profile')}": row.get("position_rmse_m") for row in rows},
    }
    return compact


def report_run(run_dir: str | Path) -> dict[str, Any]:
    run = Path(run_dir)
    summary = read_json(run / "summary.json")
    report = write_report(run, summary)
    _write_manifest(run, summary)
    return {"report": str(report), "result_count": len(summary.get("results", []))}


def export_matlab_reference(config_path: str | Path, out_dir: str | Path) -> dict[str, Any]:
    config = load_config(config_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    truth, measurements = simulate_navigation(config)
    write_csv(out / "truth_reference.csv", truth)
    write_csv(out / "measurement_reference.csv", measurements)
    write_json(out / "config_reference.json", config)
    write_json(out / "reference_manifest.json", {"truth_rows": len(truth), "measurement_rows": len(measurements), "sample_rate_hz": config["truth_sample_rate_hz"]})
    return {"out": str(out), "truth_rows": len(truth), "measurement_rows": len(measurements)}


def list_example_configs(config_dir: str | Path = "examples/configs") -> dict[str, Any]:
    directory = Path(config_dir)
    examples: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            config, errors = validate_config_file(path)
            examples.append(
                {
                    "path": str(path),
                    "name": config.get("name", path.stem),
                    "valid": not errors,
                    "errors": errors,
                    "duration_s": config.get("duration_s"),
                    "filters": config.get("estimator", {}).get("filters", []),
                    "profiles": config.get("estimator", {}).get("profiles", []),
                    "experiment_count": len(config.get("experiments", [])),
                }
            )
        except Exception as exc:
            examples.append({"path": str(path), "name": path.stem, "valid": False, "errors": [str(exc)], "experiment_count": 0})
    return {"config_dir": str(directory), "count": len(examples), "examples": examples}


def validate_dataset(input_path: str | Path, jsonl_out: str | Path | None = None) -> dict[str, Any]:
    path = Path(input_path)
    events = read_jsonl(path) if path.suffix == ".jsonl" else read_event_csv(path)
    errors = validate_events(events)
    summary = summarize_events(events)
    if jsonl_out is not None:
        write_jsonl(jsonl_out, events)
        summary["jsonl_out"] = str(jsonl_out)
    return {"valid": not errors, "errors": errors, **summary}


def analyze_residual_file(input_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    rows = read_event_csv(input_path)
    stats = estimate_noise_from_innovations(rows)
    residual_norms = np.array([float(row.get("residual_norm", 0.0)) for row in rows if row.get("residual_norm", "") != ""], dtype=float)
    stats["residual_norm_autocorrelation"] = residual_autocorrelation(residual_norms, max_lag=20)
    stats["input"] = str(input_path)
    if out_path is not None:
        write_json(Path(out_path), stats)
    return stats


def allan_variance_file(input_path: str | Path, column: str, sample_rate_hz: float, out_path: str | Path | None = None) -> dict[str, Any]:
    rows = read_event_csv(input_path)
    values = [float(row[column]) for row in rows if column in row and row[column] not in {"", None}]
    result = allan_deviation(np.asarray(values, dtype=float), sample_rate_hz)
    result.update({"input": str(input_path), "column": column, "sample_rate_hz": float(sample_rate_hz), "sample_count": len(values)})
    if out_path is not None:
        write_json(Path(out_path), result)
    return result


def analyze_fault_file(input_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    rows = read_event_csv(input_path)
    result = analyze_faults(rows)
    result["input"] = str(input_path)
    if out_path is not None:
        write_json(Path(out_path), result)
    return result


def plan_experiment_file(config_path: str | Path, out_path: str | Path | None = None) -> dict[str, Any]:
    config = load_config(config_path)
    result = plan_experiment(config)
    result["input"] = str(config_path)
    if out_path is not None:
        write_json(Path(out_path), result)
    return result


def export_c_header(run_dir: str | Path, out_path: str | Path, max_rows: int | None = None) -> dict[str, Any]:
    return export_run_c_header(run_dir, out_path, max_rows=max_rows)
