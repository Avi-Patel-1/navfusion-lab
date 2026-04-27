# Example Configurations

The `examples/configs/` directory includes:

- `ekf_imu_gps_baro.json`: baseline IMU/GPS/barometer scenario.
- `gps_dropout.json`: extended GPS outages and recovery.
- `gps_outlier_burst.json`: valid GPS samples with injected position faults.
- `high_imu_bias.json`: high accelerometer bias and bias random walk.
- `async_sensor_rates.json`: mismatched rates, GPS latency, and wheel odometry.
- `range_beacon_ekf.json`: range-only aiding with barometer.
- `magnetometer_disturbance.json`: heading disturbance during GPS dropout.
- `ukf_nonlinear_measurement.json`: EKF and UKF comparison with range beacons.
- `filter_comparison.json`: linear KF, EKF, and UKF comparison.
- `advanced_filter_comparison.json`: square-root, information, robust, adaptive, particle, linear, and EKF comparison.
- `alpha_beta_gamma_baseline.json`: smooth position-aided tracking baseline against the linear KF.
- `radar_doppler_aiding.json`: radar-altimeter and Doppler-velocity aiding through GPS dropout.
- `gnss_pseudorange_ekf.json`: raw pseudorange, barometer, and Doppler aiding without Cartesian GPS fixes.
- `multi_experiment.json`: one config that expands into baseline, dropout, and outlier experiments.

Each config can be validated with:

```bash
python3 -m fusion_sandbox validate-config --config examples/configs/<name>.json
```

To discover all bundled examples and their validation status:

```bash
python3 -m fusion_sandbox list-examples
```
