# Sensor Models

Each sensor writes timestamped measurement events with `time_s`, `measurement_time_s`, `sensor`, `kind`, `valid`, `is_dropout`, and `is_outlier` fields. Delayed measurements use a later `time_s` while preserving the original `measurement_time_s`.

## IMU

The IMU produces acceleration and gyro samples. Acceleration includes scale factor, fixed bias, bias random walk, and white noise. Gyro yaw rate is derived from the truth heading and includes white noise plus fixed bias.

## GPS

GPS emits position and velocity at its configured rate. Dropout windows keep samples in the log with `valid = 0`; outlier bursts add a configurable position offset while leaving the sample valid so gating can reject it.

## Barometer

The barometer emits altitude with white noise and a slowly varying bias. It constrains the vertical channel during GPS dropouts.

## Magnetometer

The magnetometer emits heading. Disturbance events add a heading offset and mark the event as an outlier for analysis.

## Range Beacon

Range beacons emit nonlinear scalar ranges to fixed beacon positions. These are useful for EKF and UKF comparisons because observability depends strongly on beacon geometry.

## Wheel Odometry

Wheel odometry emits planar velocity. It is optional and supports ground-navigation examples or asynchronous-rate tests.

## Radar Altimeter

The radar altimeter emits terrain-relative altitude. It supports warmup behavior, packet loss, timestamp jitter, dropout windows, fixed bias, terrain offset, and altitude outlier bursts. The filter converts AGL measurements back to vertical position by adding the configured terrain altitude.

## Doppler Velocity

The Doppler velocity source emits 3D velocity. It supports scale factor, small-axis misalignment, packet loss, timestamp jitter, warmup, dropout windows, and velocity outlier bursts. It is useful for GPS-denied velocity aiding examples.

## GNSS Pseudorange

The GNSS pseudorange source emits raw scalar ranges to configured satellite positions. It supports receiver clock bias, optional clock-bias correction, clock drift, packet loss, timestamp jitter, dropout windows, and pseudorange outlier bursts. EKF-style filters consume the measurements as nonlinear ranges to satellite positions.

## Dataset Events

Measurement CSV and JSONL replay use the same event dictionary shape as simulated sensors. `validate-dataset` checks required fields, monotonic timestamps, validity flags, and per-sensor counts before a dataset is used for analysis.
