import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

from fusion_sandbox.analysis.metrics import compute_metrics
from fusion_sandbox.calibration.noise import allan_deviation, estimate_noise_from_innovations, residual_autocorrelation
from fusion_sandbox.config import normalize_config, validate_config
from fusion_sandbox.datasets.events import read_jsonl, summarize_events, validate_events, write_jsonl
from fusion_sandbox.experiments.manager import expand_scenario_matrix, plan_experiment
from fusion_sandbox.fdir.health import SensorHealthMonitor, analyze_faults, whiteness_p_value_proxy
from fusion_sandbox.filters import PROFILES, NavFilter, create_filter
from fusion_sandbox.filters.smoothing import FixedLagSmoother, fixed_lag_smoother, rts_smoother
from fusion_sandbox.frames import euler_to_quaternion, quaternion_to_dcm, quaternion_to_euler
from fusion_sandbox.math_utils import numerical_jacobian, wrap_angle
from fusion_sandbox.model import simulate_measurements, simulate_navigation
from fusion_sandbox.navigation import (
    ComplementaryAttitudeFilter,
    IMUPreintegrator,
    MadgwickAttitudeFilter,
    MahonyAttitudeFilter,
    StrapdownState,
    ecef_to_enu,
    ecef_to_geodetic,
    ecef_to_ned,
    enu_to_ecef,
    geodetic_to_ecef,
    gravity_normal,
    mechanize,
    ned_to_ecef,
    propagate_quaternion,
)
from fusion_sandbox.pipeline import (
    allan_variance_file,
    analyze_fault_file,
    analyze_residual_file,
    export_matlab_reference,
    export_c_header,
    list_example_configs,
    plan_experiment_file,
    report_run,
    run_config,
    run_experiment,
    validate_dataset,
)
from fusion_sandbox.sensors.faults import in_windows
from fusion_sandbox.sensors.timing import sample_times
from fusion_sandbox.trajectory import generate_truth


ROOT = Path(__file__).resolve().parents[1]


def short_config(**overrides):
    cfg = normalize_config(
        {
            "name": "unit_short",
            "seed": 7,
            "duration_s": 6.0,
            "truth_sample_rate_hz": 10.0,
            "trajectory": {"profile": "mixed", "mode": "3d"},
            "sensors": {
                "imu": {"rate_hz": 10.0},
                "gps": {
                    "rate_hz": 1.0,
                    "dropout_windows_s": [[2.0, 3.0]],
                    "outlier_bursts": [{"start_s": 4.0, "duration_s": 0.2, "position_offset_m": [80.0, -50.0, 20.0]}],
                },
                "barometer": {"rate_hz": 2.0},
            },
            "estimator": {"filters": ["ekf"], "profiles": ["nominal"], "max_stale_s": 1.0},
        }
    )
    for key, value in overrides.items():
        cfg[key] = value
    return cfg


class ConfigAndTrajectoryTests(unittest.TestCase):
    def test_config_validation_accepts_baseline(self):
        cfg = normalize_config(json.loads((ROOT / "examples/configs/ekf_imu_gps_baro.json").read_text()))
        self.assertEqual(validate_config(cfg), [])
        self.assertIn("nominal", cfg["estimator"]["profiles"])

    def test_config_validation_rejects_bad_duration(self):
        cfg = short_config()
        cfg["duration_s"] = -1.0
        self.assertTrue(any("duration" in err for err in validate_config(cfg)))

    def test_config_validation_rejects_negative_noise_and_bad_beacons(self):
        cfg = short_config()
        cfg["sensors"]["gps"]["position_noise_std_m"] = -0.1
        cfg["sensors"]["range_beacon"]["enabled"] = True
        cfg["sensors"]["range_beacon"]["beacons_m"] = []
        errors = validate_config(cfg)
        self.assertTrue(any("position_noise" in err for err in errors))
        self.assertTrue(any("beacons_m" in err for err in errors))

    def test_config_validation_checks_new_sensor_fields(self):
        cfg = short_config()
        cfg["sensors"]["radar_altimeter"]["enabled"] = True
        cfg["sensors"]["radar_altimeter"]["packet_loss_probability"] = 1.2
        cfg["sensors"]["doppler_velocity"]["enabled"] = True
        cfg["sensors"]["doppler_velocity"]["scale_factor"] = [1.0, 1.0]
        cfg["sensors"]["gnss_pseudorange"]["enabled"] = True
        cfg["sensors"]["gnss_pseudorange"]["satellites_m"] = [[1.0, 2.0, 3.0]]
        errors = validate_config(cfg)
        self.assertTrue(any("radar_altimeter.packet_loss_probability" in err for err in errors))
        self.assertTrue(any("doppler_velocity.scale_factor" in err for err in errors))
        self.assertTrue(any("gnss_pseudorange.satellites_m" in err for err in errors))

    def test_config_validation_checks_experiment_overrides(self):
        cfg = short_config()
        cfg["experiments"] = [
            {"name": "bad_override", "overrides": {"sensors": {"gps": {"position_noise_std_m": -1.0}}}}
        ]
        errors = validate_config(cfg)
        self.assertTrue(any("experiments[0].overrides" in err for err in errors))

    def test_trajectory_generation_2d_and_figure_eight(self):
        cfg = short_config()
        cfg["trajectory"]["mode"] = "2d"
        truth = generate_truth(cfg)
        self.assertTrue(all(abs(row["pz"]) < 1e-12 for row in truth))
        cfg["trajectory"]["mode"] = "3d"
        cfg["trajectory"]["profile"] = "figure_eight"
        truth = generate_truth(cfg)
        self.assertGreater(np.ptp([row["px"] for row in truth]), 1.0)

    def test_frame_utilities_round_trip(self):
        q = euler_to_quaternion(0.1, -0.2, 0.3)
        roll, pitch, yaw = quaternion_to_euler(q)
        self.assertAlmostEqual(roll, 0.1, places=6)
        self.assertAlmostEqual(pitch, -0.2, places=6)
        self.assertAlmostEqual(yaw, 0.3, places=6)
        dcm = quaternion_to_dcm(q)
        self.assertTrue(np.allclose(dcm @ dcm.T, np.eye(3), atol=1e-9))


class SensorTests(unittest.TestCase):
    def test_measurement_simulation_compatibility(self):
        truth, measurements = simulate_measurements({"duration_s": 2.0, "dt_s": 0.1})
        self.assertEqual(len(truth), len(measurements))
        self.assertIn("imu_ax", measurements[0])

    def test_sensor_logs_include_dropout_and_outlier_flags(self):
        _, events = simulate_navigation(short_config())
        gps = [row for row in events if row["sensor"] == "gps"]
        self.assertTrue(any(row["valid"] == 0.0 and row["is_dropout"] == 1.0 for row in gps))
        self.assertTrue(any(row["is_outlier"] == 1.0 for row in gps))
        self.assertTrue(in_windows(2.5, [[2.0, 3.0]]))

    def test_sample_times_can_include_endpoint(self):
        regular = sample_times(1.0, 2.5)
        with_endpoint = sample_times(1.0, 2.5, include_endpoint=True)
        self.assertNotEqual(regular[-1], 1.0)
        self.assertEqual(with_endpoint[-1], 1.0)

    def test_optional_sensor_models_emit_events(self):
        cfg = short_config()
        cfg["sensors"]["magnetometer"]["enabled"] = True
        cfg["sensors"]["range_beacon"]["enabled"] = True
        cfg["sensors"]["wheel_odometry"]["enabled"] = True
        cfg["sensors"]["radar_altimeter"]["enabled"] = True
        cfg["sensors"]["radar_altimeter"]["warmup_s"] = 0.2
        cfg["sensors"]["doppler_velocity"]["enabled"] = True
        cfg["sensors"]["doppler_velocity"]["outlier_bursts"] = [{"start_s": 1.0, "duration_s": 0.3, "velocity_offset_mps": [1.0, 0.0, 0.0]}]
        cfg["sensors"]["gnss_pseudorange"]["enabled"] = True
        cfg["sensors"]["gnss_pseudorange"]["rate_hz"] = 1.0
        cfg["sensors"]["gnss_pseudorange"]["outlier_bursts"] = [{"start_s": 1.0, "duration_s": 0.2, "pseudorange_offset_m": 40.0}]
        _, events = simulate_navigation(cfg)
        sensors = {row["sensor"] for row in events}
        self.assertIn("magnetometer", sensors)
        self.assertIn("range_beacon", sensors)
        self.assertIn("wheel_odometry", sensors)
        self.assertIn("radar_altimeter", sensors)
        self.assertIn("doppler_velocity", sensors)
        self.assertIn("gnss_pseudorange", sensors)
        radar = [row for row in events if row["sensor"] == "radar_altimeter"]
        doppler = [row for row in events if row["sensor"] == "doppler_velocity"]
        gnss = [row for row in events if row["sensor"] == "gnss_pseudorange"]
        self.assertTrue(any(row["valid"] == 0.0 and row["is_dropout"] == 1.0 for row in radar))
        self.assertTrue(any(row["is_outlier"] == 1.0 for row in doppler))
        self.assertTrue(any(row["is_outlier"] == 1.0 for row in gnss))


class NavigationAndCalibrationTests(unittest.TestCase):
    def test_wgs84_geodetic_ecef_and_local_frames_round_trip(self):
        ref = (np.deg2rad(42.3601), np.deg2rad(-71.0589), 34.0)
        ecef = geodetic_to_ecef(*ref)
        lat, lon, alt = ecef_to_geodetic(ecef)
        self.assertAlmostEqual(lat, ref[0], places=10)
        self.assertAlmostEqual(lon, ref[1], places=10)
        self.assertAlmostEqual(alt, ref[2], places=4)
        enu = np.array([12.0, -4.0, 2.5])
        self.assertTrue(np.allclose(ecef_to_enu(enu_to_ecef(enu, ref), ref), enu, atol=1e-6))
        ned = np.array([5.0, 7.0, -1.5])
        self.assertTrue(np.allclose(ecef_to_ned(ned_to_ecef(ned, ref), ref), ned, atol=1e-6))
        self.assertGreater(gravity_normal(ref[0], ref[2]), 9.79)

    def test_strapdown_mechanization_and_preintegration(self):
        state = StrapdownState(
            position_m=np.zeros(3),
            velocity_mps=np.zeros(3),
            q_nav_from_body=np.array([1.0, 0.0, 0.0, 0.0]),
            gyro_bias_radps=np.zeros(3),
            accel_bias_mps2=np.zeros(3),
        )
        propagated = mechanize(state, np.array([0.0, 0.0, 9.80665]), np.zeros(3), 0.1)
        self.assertTrue(np.allclose(propagated.velocity_mps, np.zeros(3), atol=1e-10))
        q = propagate_quaternion(state.q_nav_from_body, np.array([0.0, 0.0, 0.2]), 0.5)
        self.assertAlmostEqual(float(np.linalg.norm(q)), 1.0, places=12)
        preint = IMUPreintegrator()
        for _ in range(5):
            preint.add_sample(np.array([1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.1]), 0.1)
        result = preint.result()
        self.assertEqual(result.sample_count, 5)
        self.assertAlmostEqual(result.dt_s, 0.5)
        self.assertGreater(result.delta_velocity_mps[0], 0.0)
        self.assertGreater(result.delta_theta_rad[2], 0.0)

    def test_attitude_filters_integrate_and_correct_tilt(self):
        comp = ComplementaryAttitudeFilter(accel_gain=0.08)
        for _ in range(20):
            est = comp.update(np.array([0.0, 0.0, np.pi / 2.0]), np.array([0.0, 0.0, 9.80665]), 0.05)
        self.assertAlmostEqual(est.yaw_rad, np.pi / 2.0, places=1)
        tilted = ComplementaryAttitudeFilter(q_nav_from_body=euler_to_quaternion(0.25, 0.0, 0.0), accel_gain=0.15)
        before = abs(tilted.estimate().roll_rad)
        for _ in range(30):
            tilted.update(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 0.02)
        self.assertLess(abs(tilted.estimate().roll_rad), before)
        mahony = MahonyAttitudeFilter(proportional_gain=0.6, integral_gain=0.02)
        madgwick = MadgwickAttitudeFilter(beta=0.05)
        for _ in range(10):
            mahony_est = mahony.update(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 0.01)
            madgwick_est = madgwick.update(np.zeros(3), np.array([0.0, 0.0, 9.80665]), 0.01)
        self.assertAlmostEqual(float(np.linalg.norm(mahony_est.q_nav_from_body)), 1.0, places=12)
        self.assertAlmostEqual(float(np.linalg.norm(madgwick_est.q_nav_from_body)), 1.0, places=12)

    def test_residual_noise_allan_and_whiteness_utilities(self):
        innovations = [
            {"residual_0": 0.2, "residual_1": -0.1, "residual_norm": 0.22},
            {"residual_0": -0.1, "residual_1": 0.05, "residual_norm": 0.11},
            {"residual_0": 0.05, "residual_1": 0.0, "residual_norm": 0.05},
        ]
        stats = estimate_noise_from_innovations(innovations)
        self.assertEqual(stats["count"], 3)
        self.assertEqual(len(stats["residual_std"]), 2)
        self.assertEqual(residual_autocorrelation(np.array([1.0, 2.0, 1.0]), max_lag=0), [1.0])
        allan = allan_deviation(np.linspace(0.0, 1.0, 64), 8.0)
        self.assertGreater(len(allan["tau_s"]), 0)
        self.assertGreaterEqual(whiteness_p_value_proxy(np.array([0.1, -0.2, 0.15, -0.1])), 0.0)


class FilterTests(unittest.TestCase):
    def test_filter_predict_changes_state(self):
        f = NavFilter(PROFILES[1], np.array([0.0, 0.0, 0.0]))
        f.predict(np.array([1.0, 0.0, 0.0]), 0.1)
        self.assertGreater(f.x[3], 0.0)

    def test_covariance_symmetry_and_positive_diagonal(self):
        f = create_filter("ekf", "nominal", np.array([10.0, 2.0, 5.0]), np.array([1.0, 0.1, 0.0]))
        f.predict(np.array([0.1, 0.0, 0.0]), 0.1)
        f.update_baro(5.2)
        self.assertTrue(np.allclose(f.P, f.P.T, atol=1e-9))
        self.assertTrue(np.all(np.diag(f.P) > 0.0))

    def test_gating_rejects_large_outlier(self):
        f = create_filter("ekf", "nominal", np.array([0.0, 0.0, 0.0]), np.zeros(3))
        rec = f.update_baro(10000.0)
        self.assertEqual(rec["accepted"], 0.0)

    def test_ekf_range_jacobian_matches_numerical(self):
        beacon = np.array([1.0, -2.0, 3.0])
        x = np.array([8.0, 4.0, 7.0])

        def h(pos):
            return np.array([np.linalg.norm(pos - beacon)])

        numeric = numerical_jacobian(h, x)
        analytic = ((x - beacon) / np.linalg.norm(x - beacon)).reshape(1, 3)
        self.assertTrue(np.allclose(numeric, analytic, atol=1e-5))

    def test_ukf_sigma_point_dimensions(self):
        f = create_filter("ukf", "nominal", np.array([0.0, 0.0, 0.0]), np.zeros(3))
        sigmas = f.sigma_points()
        self.assertEqual(sigmas.shape, (19, 9))
        f.predict(np.array([0.0, 0.0, 0.0]), 0.1)
        self.assertTrue(np.all(np.diag(f.P) > 0.0))

    def test_advanced_filter_families_have_predict_update_api(self):
        for filter_type in ["sqrt", "information", "robust", "adaptive", "abg", "particle"]:
            with self.subTest(filter_type=filter_type):
                f = create_filter(filter_type, "nominal", np.array([1.0, 2.0, 3.0]), np.array([0.5, -0.2, 0.1]))
                f.predict(np.array([0.05, -0.01, 0.0]), 0.2)
                rec = f.update_gps(np.array([1.2, 1.9, 3.05]), np.array([0.45, -0.18, 0.08]))
                self.assertIn("accepted", rec)
                self.assertTrue(np.all(np.isfinite(f.x)))
                self.assertTrue(np.all(np.isfinite(f.P)))
                self.assertTrue(np.all(np.diag(f.P) > 0.0))

    def test_radar_altimeter_and_doppler_updates(self):
        f = create_filter("ekf", "nominal", np.array([0.0, 0.0, 100.0]), np.array([5.0, 0.0, -0.2]))
        f.predict(np.zeros(3), 0.1)
        radar = f.update_radar_altimeter(98.2, terrain_altitude_m=2.0)
        doppler = f.update_doppler_velocity(np.array([5.1, 0.1, -0.1]))
        self.assertEqual(radar["accepted"], 1.0)
        self.assertEqual(doppler["accepted"], 1.0)
        self.assertIn("radar", radar["source"])
        self.assertIn("doppler", doppler["source"])

    def test_gnss_pseudorange_update(self):
        sat = np.array([15600000.0, 7540000.0, 20140000.0])
        f = create_filter("ekf", "nominal", np.array([10.0, -5.0, 100.0]), np.array([2.0, 0.0, 0.0]))
        pseudorange = float(np.linalg.norm(f.x[:3] - sat)) + 0.5
        rec = f.update_gnss_pseudorange(pseudorange, sat, clock_bias_correction_m=0.0)
        self.assertEqual(rec["accepted"], 1.0)
        self.assertIn("gnss", rec["source"])

    def test_huber_update_reduces_outlier_influence(self):
        robust = create_filter("robust", "nominal", np.zeros(3), np.zeros(3))
        linear = create_filter("linear", "nominal", np.zeros(3), np.zeros(3))
        z = np.array([25.0, 0.0, 0.0])
        robust.update_gps(z, np.zeros(3))
        linear.update_gps(z, np.zeros(3))
        self.assertLess(abs(robust.x[0]), abs(linear.x[0]))

    def test_rts_smoother_shapes_and_covariance(self):
        states = np.array([[0.0, 1.0], [1.1, 1.0], [2.0, 1.0]])
        covariances = np.repeat(np.eye(2)[None, :, :] * 0.5, 3, axis=0)
        transitions = np.repeat(np.array([[[1.0, 1.0], [0.0, 1.0]]]), 2, axis=0)
        process = np.repeat(np.eye(2)[None, :, :] * 0.05, 2, axis=0)
        smooth_states, smooth_covs = rts_smoother(states, covariances, transitions, process)
        self.assertEqual(smooth_states.shape, states.shape)
        self.assertEqual(smooth_covs.shape, covariances.shape)
        self.assertTrue(np.all(np.diag(smooth_covs[0]) > 0.0))

    def test_fixed_lag_smoother_batch_and_streaming(self):
        states = np.array([[0.0, 1.0], [1.2, 1.0], [2.1, 1.0], [3.0, 1.0]])
        covariances = np.repeat(np.eye(2)[None, :, :] * 0.4, 4, axis=0)
        transitions = np.repeat(np.array([[[1.0, 1.0], [0.0, 1.0]]]), 3, axis=0)
        process = np.repeat(np.eye(2)[None, :, :] * 0.04, 3, axis=0)
        xs, Ps = fixed_lag_smoother(states, covariances, transitions, process, lag=2)
        self.assertEqual(xs.shape, states.shape)
        self.assertEqual(Ps.shape, covariances.shape)
        self.assertTrue(np.all(np.isfinite(xs)))
        self.assertTrue(np.all(np.diag(Ps[0]) > 0.0))
        smoother = FixedLagSmoother(lag=1)
        self.assertIsNone(smoother.add(states[0], covariances[0]))
        released = smoother.add(states[1], covariances[1], transitions[0], process[0])
        self.assertIsNone(released)
        released = smoother.add(states[2], covariances[2], transitions[1], process[1])
        self.assertIsNotNone(released)
        self.assertGreaterEqual(len(smoother.flush()), 1)

    def test_angle_wrapping(self):
        self.assertAlmostEqual(float(wrap_angle(3.5)), -2.7831853071795862)


class PipelineAndCliTests(unittest.TestCase):
    def test_experiment_outputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_experiment(ROOT / "examples/configs/ekf_imu_gps_baro.json", tmp)
            self.assertIn("best_profile", summary)
            self.assertTrue((Path(tmp) / "summary.json").exists())
            self.assertTrue((Path(tmp) / "metrics_ekf_nominal.json").exists())
            self.assertTrue((Path(tmp) / "manifest.json").exists())
            self.assertTrue((Path(tmp) / "plots/position_error_ekf_nominal.svg").exists())
            self.assertTrue((Path(tmp) / "report.html").exists())

    def test_short_pipeline_outputs_and_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_config(short_config(), tmp)
            self.assertEqual(summary["samples"], 61)
            metrics = json.loads((Path(tmp) / "metrics_ekf_nominal.json").read_text())
            self.assertGreater(metrics["position_rmse_m"], 0.0)
            self.assertIn("updates_by_sensor", metrics)

    def test_run_config_accepts_partial_dict(self):
        with tempfile.TemporaryDirectory() as tmp:
            summary = run_config({"name": "partial", "duration_s": 1.0}, tmp)
            self.assertEqual(summary["name"], "partial")
            self.assertTrue((Path(tmp) / "config_resolved.json").exists())

    def test_metrics_function_direct(self):
        truth = [{"time_s": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0, "vx": 0.0, "vy": 0.0, "vz": 0.0}]
        estimates = [
            {
                "time_s": 0.0,
                "px": 1.0,
                "py": 0.0,
                "pz": 0.0,
                "vx": 0.0,
                "vy": 0.0,
                "vz": 0.0,
                "covariance_trace": 1.0,
                "nees_position": 1.0,
            }
        ]
        metrics = compute_metrics(estimates, [], truth, [], short_config())
        self.assertAlmostEqual(metrics["position_rmse_m"], 1.0)

    def test_metrics_interpolate_truth_to_estimate_times(self):
        truth = [
            {"time_s": 0.0, "px": 0.0, "py": 0.0, "pz": 0.0, "vx": 1.0, "vy": 0.0, "vz": 0.0},
            {"time_s": 1.0, "px": 1.0, "py": 0.0, "pz": 0.0, "vx": 1.0, "vy": 0.0, "vz": 0.0},
        ]
        estimates = [
            {"time_s": 0.5, "px": 0.5, "py": 0.0, "pz": 0.0, "vx": 1.0, "vy": 0.0, "vz": 0.0, "covariance_trace": 1.0, "nees_position": 0.0}
        ]
        metrics = compute_metrics(estimates, [], truth, [], short_config())
        self.assertAlmostEqual(metrics["position_rmse_m"], 0.0)

    def test_nondivisor_estimator_rate_writes_final_sample(self):
        cfg = short_config()
        cfg["duration_s"] = 5.0
        cfg["estimator"]["update_rate_hz"] = 3.0
        with tempfile.TemporaryDirectory() as tmp:
            run_config(cfg, tmp)
            estimate_rows = (Path(tmp) / "estimates_ekf_nominal.csv").read_text().strip().splitlines()
            last = estimate_rows[-1].split(",")
            header = estimate_rows[0].split(",")
            self.assertAlmostEqual(float(last[header.index("time_s")]), 5.0)

    def test_report_and_matlab_export(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_config(short_config(), Path(tmp) / "run")
            report = report_run(Path(tmp) / "run")
            self.assertTrue(Path(report["report"]).exists())
            exported = export_matlab_reference(ROOT / "examples/configs/ekf_imu_gps_baro.json", Path(tmp) / "matlab_ref")
            self.assertGreater(exported["truth_rows"], 0)
            self.assertTrue((Path(tmp) / "matlab_ref/truth_reference.csv").exists())

    def test_list_examples_reports_valid_configs(self):
        listed = list_example_configs(ROOT / "examples/configs")
        self.assertGreaterEqual(listed["count"], 14)
        self.assertTrue(all(row["valid"] for row in listed["examples"]))

    def test_dataset_residual_fault_and_experiment_helpers(self):
        events = [
            {"time_s": 0.0, "sensor": "gps", "valid": 1.0, "x": 0.0, "is_outlier": 0.0, "is_dropout": 0.0},
            {"time_s": 0.5, "sensor": "gps", "valid": 0.0, "x": 1.0, "is_outlier": 1.0, "is_dropout": 0.0},
        ]
        self.assertEqual(validate_events(events), [])
        self.assertEqual(summarize_events(events)["by_sensor"]["gps"]["invalid"], 1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            jsonl_path = tmp_path / "events.jsonl"
            write_jsonl(jsonl_path, events)
            self.assertEqual(len(read_jsonl(jsonl_path)), 2)
            csv_path = tmp_path / "events.csv"
            csv_path.write_text(
                "time_s,sensor,valid,x,is_outlier,is_dropout\n"
                "0.0,gps,1,0,0,0\n"
                "0.5,gps,0,1,1,0\n"
            )
            self.assertTrue(validate_dataset(csv_path)["valid"])
            innovations = tmp_path / "innovations.csv"
            innovations.write_text(
                "time_s,source,nis,gate,accepted,residual_norm,residual_0,sensor_is_outlier,sensor_is_dropout\n"
                "0.0,gps_position,0.2,55,1,0.2,0.2,0,0\n"
                "1.0,gps_position,90.0,55,0,10.0,10.0,1,0\n"
                "2.0,barometer_altitude,0.1,55,1,0.1,0.1,0,0\n"
            )
            self.assertEqual(analyze_residual_file(innovations)["count"], 3)
            self.assertIn("gps", analyze_fault_file(innovations)["sensor_summary"])
            samples = tmp_path / "samples.csv"
            samples.write_text("time_s,sensor,valid,ax\n" + "\n".join(f"{i * 0.1},imu,1,{0.01 * i}" for i in range(32)) + "\n")
            self.assertGreater(allan_variance_file(samples, "ax", 10.0)["sample_count"], 0)
            planned = plan_experiment_file(ROOT / "examples/configs/multi_experiment.json")
            self.assertGreaterEqual(planned["scenario_count"], 3)

    def test_c_header_export_for_run_vectors(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_config({"name": "tiny_header", "duration_s": 1.0, "estimator": {"filters": ["ekf"], "profiles": ["nominal"]}}, run_dir)
            header = Path(tmp) / "vectors.h"
            exported = export_c_header(run_dir, header, max_rows=5)
            text = header.read_text()
            self.assertGreaterEqual(exported["array_count"], 3)
            self.assertIn("static const double truth", text)
            self.assertIn("FUSION_SANDBOX_REFERENCE_VECTORS_H", text)

    def test_fdir_health_monitor_quarantines_and_recovers(self):
        monitor = SensorHealthMonitor(quarantine_threshold=0.6, recovery_threshold=0.7, recovery_count=2)
        timeline = [monitor.update("gps_position", 100.0, 3, False, i) for i in range(5)]
        self.assertTrue(any(row["quarantined"] for row in timeline))
        recovery = [monitor.update("gps_position", 0.1, 3, True, 10.0 + i) for i in range(8)]
        self.assertFalse(recovery[-1]["quarantined"])
        analysis = analyze_faults(
            [
                {"time_s": 0.0, "source": "gps_position", "nis": 0.1, "accepted": 1.0, "residual_norm": 0.1},
                {"time_s": 1.0, "source": "gps_position", "nis": 100.0, "accepted": 0.0, "residual_norm": 10.0, "sensor_is_outlier": 1.0},
            ]
        )
        self.assertEqual(analysis["sensor_summary"]["gps"]["truth_fault_labels"], 1)
        self.assertEqual(analysis["sensor_summary"]["gps"]["confusion_matrix"]["tp"], 1)
        self.assertEqual(analysis["sensor_summary"]["gps"]["detection_latency_s"], 0.0)

    def test_experiment_matrix_hashes_are_deterministic(self):
        cfg = short_config()
        scenarios = expand_scenario_matrix(cfg, {"sensors.gps.position_noise_std_m": [1.0, 2.0], "estimator.profiles": [["nominal"]]})
        self.assertEqual(len(scenarios), 2)
        first = plan_experiment({"name": "matrix", "experiments": [{"name": "a", "overrides": {}}], **cfg})
        second = plan_experiment({"name": "matrix", "experiments": [{"name": "a", "overrides": {}}], **cfg})
        self.assertEqual(first["scenarios"][0]["hash"], second["scenarios"][0]["hash"])

    def test_cli_smoke(self):
        result = subprocess.run(
            [sys.executable, "-m", "fusion_sandbox", "validate-config", "--config", "examples/configs/ekf_imu_gps_baro.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"valid": true', result.stdout)
        result = subprocess.run(
            [sys.executable, "-m", "fusion_sandbox", "list-examples"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"examples"', result.stdout)
        result = subprocess.run(
            [sys.executable, "-m", "fusion_sandbox", "plan-experiment", "--config", "examples/configs/multi_experiment.json"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        )
        self.assertIn('"scenario_count"', result.stdout)
        with tempfile.TemporaryDirectory() as tmp:
            dataset = Path(tmp) / "events.csv"
            dataset.write_text("time_s,sensor,valid\n0.0,gps,1\n")
            result = subprocess.run(
                [sys.executable, "-m", "fusion_sandbox", "validate-dataset", "--input", str(dataset)],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"valid": true', result.stdout)
            run_dir = Path(tmp) / "run"
            run_config({"name": "cli_header", "duration_s": 0.5, "estimator": {"filters": ["ekf"], "profiles": ["nominal"]}}, run_dir)
            header = Path(tmp) / "vectors.h"
            result = subprocess.run(
                [sys.executable, "-m", "fusion_sandbox", "export-c-header", "--run", str(run_dir), "--out", str(header), "--max-rows", "3"],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=True,
            )
            self.assertIn('"array_count"', result.stdout)
            self.assertTrue(header.exists())

    def test_matlab_files_present(self):
        required = [
            "ekf_predict.m",
            "ekf_update_gps.m",
            "ekf_update_baro.m",
            "ukf_predict_update.m",
            "generate_reference_data.m",
            "run_demo.m",
            "README.md",
        ]
        for name in required:
            self.assertTrue((ROOT / "matlab" / name).exists(), name)


if __name__ == "__main__":
    unittest.main()
