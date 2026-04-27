# Simulink Reconstruction Guide

This guide describes a Simulink model that can reproduce the Python sandbox experiments and compare against the CSV reference data.

## Top-Level Model

```mermaid
flowchart LR
  Truth["Trajectory Source"] --> Sensors["Sensor Models"]
  Sensors --> Align["Time Alignment and Validity"]
  Align --> Estimator["Navigation Estimator"]
  Estimator --> Metrics["Metrics and Scopes"]
  Truth --> Metrics
```

## Subsystems

1. **Trajectory Source**
   - Outputs position, velocity, inertial acceleration, and yaw in a local tangent frame.
   - Use From Workspace blocks loaded from `matlab/reference/truth_reference.csv`.
   - Sample time should match `truth_sample_rate_hz` in the resolved config.

2. **Sensor Models**
   - IMU: acceleration plus bias, scale factor, white noise, gyro yaw rate, and bias random walk.
   - GPS: position/velocity samples, validity flag, dropout flag, outlier flag, and latency.
   - Barometer: altitude samples with slowly varying bias.
   - Magnetometer: heading measurement with disturbance flag.
   - Range beacon: one range per beacon geometry.
   - Wheel odometry: planar velocity source for ground-style examples.

3. **Time Alignment and Validity**
   - Route only valid measurements to update blocks.
   - Compare `time_s - measurement_time_s` with `max_stale_s`.
   - Hold latest IMU between estimator ticks.
   - Drop delayed GPS if it exceeds the stale threshold, or log it as rejected.

4. **Estimator**
   - Predict: `matlab/ekf_predict.m`.
   - GPS Cartesian update: `matlab/ekf_update_gps.m`.
   - GPS range/bearing/altitude update: `matlab/ekf_update_gps_range_bearing.m`.
   - Barometer update: `matlab/ekf_update_baro.m`.
   - UKF reference: `matlab/ukf_predict_update.m`.
   - Gating: compare NIS against the selected profile threshold.

5. **Metrics and Scopes**
   - Position and velocity error norms.
   - NIS and accepted/rejected flags.
   - `sqrt(trace(P_pos))` and full covariance trace.
   - Bias estimates and bias error.
   - Dropout window maximum error and recovery time.

## Recommended Buses

`truth_bus`

- `time_s`
- `position_m[3]`
- `velocity_mps[3]`
- `acceleration_mps2[3]`
- `yaw_rad`

`imu_bus`

- `time_s`
- `accel_mps2[3]`
- `gyro_radps[3]`
- `valid`

`gps_bus`

- `time_s`
- `measurement_time_s`
- `position_m[3]`
- `velocity_mps[3]`
- `valid`
- `is_dropout`
- `is_outlier`

`estimate_bus`

- `time_s`
- `state[9]`
- `covariance_diag[9]`
- `accepted_update`
- `last_source`
- `nis`

## Sample Times

- Estimator: `1 / estimator.update_rate_hz`.
- IMU: usually equal to the estimator rate.
- GPS: typically `1 s` or slower.
- Barometer: typically `0.2 s`.
- Range beacon and magnetometer: scenario dependent.

Use a fixed-step discrete solver. If source sample times are asynchronous, let each source run at its own discrete sample time and route through the Time Alignment subsystem.

## Validation Procedure

1. Run `python3 -m fusion_sandbox export-matlab --config examples/configs/ekf_imu_gps_baro.json --out matlab/reference`.
2. Load `truth_reference.csv` and `measurement_reference.csv` into MATLAB timetables.
3. Drive the model from those timetables.
4. Save estimate and innovation logs from Simulink.
5. Compare RMSE, accepted/rejected counts, and final covariance trace against `outputs/<run>/summary.json`.
6. Confirm covariance symmetry after every update.

## Reconstruction Checklist

- Use the same initial estimate offset as `config_resolved.json`.
- Match process and measurement noise from the selected tuning profile.
- Apply Joseph-form covariance updates.
- Wrap bearing and heading residuals to `[-pi, pi)`.
- Reject measurements whose NIS exceeds the profile gate.
- Inflate covariance during configured GPS dropout windows.
- Keep invalid dropout samples in the log for auditability, but do not update the state with them.
