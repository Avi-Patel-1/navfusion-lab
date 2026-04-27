# State Estimation and Sensor Fusion Sandbox

A NumPy-only navigation-estimation sandbox for designing, tuning, and validating sensor-fusion filters. It simulates truth trajectories, asynchronous sensors, dropout windows, delayed measurements, outlier bursts, and multiple Kalman-filter families, then writes CSV, JSON, SVG, HTML, MATLAB, and Simulink handoff artifacts.

## What It Covers

- 2D and 3D truth profiles: straight line, coordinated turn, climb/descent, aggressive acceleration, low-speed hover, figure-eight, and mixed maneuvers.
- Sensor models: IMU, GPS, barometer, magnetometer, range beacons, wheel odometry, radar altimeter, Doppler velocity, and GNSS pseudorange.
- Fault modeling: GPS dropout, GPS outliers, magnetometer disturbances, sensor latency, and stale measurements.
- Filters: linear KF, square-root KF, information filter, EKF, UKF, robust Huber update, innovation-adaptive update, particle filter, and alpha-beta-gamma baseline.
- Navigation utilities: WGS84 geodetic/ECEF/ENU/NED transforms, normal gravity, quaternion propagation, strapdown mechanization, and IMU preintegration.
- Tuning profiles: aggressive, nominal, conservative, dropout robust, and high bias.
- Analysis: RMSE, NIS, NEES proxy, residual statistics, Allan deviation, sensor health timelines, dropout recovery, accepted/rejected counts, covariance trace, and divergence flag.
- Reports: per-run CSV logs, JSON metrics, SVG plots, and `report.html`.

## Quick Start

```bash
cd "/Users/avipatel/Documents/New project 3/state_estimation_sensor_fusion_sandbox"
python3 -m unittest discover -s tests
python3 -m fusion_sandbox validate-config --config examples/configs/ekf_imu_gps_baro.json
python3 -m fusion_sandbox list-examples
python3 -m fusion_sandbox run --config examples/configs/ekf_imu_gps_baro.json --out outputs/demo
```

## CLI

```bash
python3 -m fusion_sandbox validate-config --config examples/configs/ekf_imu_gps_baro.json
python3 -m fusion_sandbox run --config examples/configs/ekf_imu_gps_baro.json --out outputs/demo
python3 -m fusion_sandbox compare --config examples/configs/filter_comparison.json --out outputs/compare
python3 -m fusion_sandbox sweep-tuning --config examples/configs/ekf_imu_gps_baro.json --out outputs/tuning_sweep
python3 -m fusion_sandbox summarize --input outputs/demo/summary.json
python3 -m fusion_sandbox export-matlab --config examples/configs/ekf_imu_gps_baro.json --out matlab/reference
python3 -m fusion_sandbox report --run outputs/demo
python3 -m fusion_sandbox validate-dataset --input outputs/demo/measurements.csv --jsonl-out outputs/demo/measurements.jsonl
python3 -m fusion_sandbox analyze-residuals --input outputs/demo/innovations_ekf_nominal.csv --out outputs/demo/residual_analysis.json
python3 -m fusion_sandbox allan-variance --input outputs/demo/measurements.csv --column ax --sample-rate-hz 20 --out outputs/demo/allan_ax.json
python3 -m fusion_sandbox analyze-faults --input outputs/demo/innovations_ekf_nominal.csv --out outputs/demo/fault_analysis.json
python3 -m fusion_sandbox plan-experiment --config examples/configs/multi_experiment.json --out outputs/plan.json
python3 -m fusion_sandbox export-c-header --run outputs/demo --out outputs/demo/reference_vectors.h
```

## Output Layout

Each run writes:

- `truth.csv`
- `measurements.csv`
- `estimates_<filter>_<profile>.csv`
- `innovations_<filter>_<profile>.csv`
- `metrics_<filter>_<profile>.json`
- `summary.json`
- `manifest.json`
- `report.html`
- `plots/*.svg`

Plots include trajectory, position error, velocity error, altitude, covariance bounds, NIS, NEES proxy, residuals, accepted/rejected counts, bias estimates, filter comparison, and tuning profile comparison.

## Project Layout

```text
fusion_sandbox/
  cli.py
  config.py
  schema.py
  frames.py
  math_utils.py
  trajectory/
  sensors/
  filters/
  navigation/
  calibration/
  datasets/
  fdir/
  experiments/
  fusion/
  analysis/
  reports/
  pipeline.py
examples/configs/
docs/
matlab/
simulink/
tests/
```

## Adding a Sensor

1. Add a module under `fusion_sandbox/sensors/`.
2. Return event dictionaries with timestamps, `sensor`, `kind`, `valid`, `is_dropout`, and `is_outlier`.
3. Add default config entries in `fusion_sandbox/config.py`.
4. Route the event in `pipeline._apply_measurement`.
5. Add tests for sampling, validity flags, and estimator behavior.

## Adding a Filter

1. Subclass `BaseKalmanFilter` under `fusion_sandbox/filters/`.
2. Implement `predict` if the process model differs.
3. Implement update methods for supported sensors.
4. Add it to `create_filter`.
5. Add a comparison config and a CLI smoke test.

## MATLAB and Simulink

Use:

```bash
python3 -m fusion_sandbox export-matlab --config examples/configs/ekf_imu_gps_baro.json --out matlab/reference
```

The MATLAB folder contains prediction and update references. The Simulink guide describes buses, sample times, scopes, and validation steps for reconstructing the estimator.
