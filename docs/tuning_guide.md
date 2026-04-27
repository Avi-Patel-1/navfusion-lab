# Tuning Guide

The built-in profiles are deliberately different:

- `aggressive`: fast convergence, lower process noise, tighter covariance.
- `nominal`: balanced process and measurement assumptions.
- `conservative`: larger measurement noise and a linear GPS update path.
- `dropout_robust`: larger process noise and stronger covariance inflation during outages.
- `high_bias`: higher accelerometer-bias random walk.

## Workflow

1. Run the baseline config.
2. Inspect position RMSE, NIS, NEES proxy, rejected counts, and dropout maximum error.
3. If NIS is consistently high, increase measurement noise or process noise.
4. If covariance trace collapses during dropout, increase dropout inflation or process noise.
5. If outliers are accepted, lower the gate or increase source-specific robustness.
6. If valid data is rejected after maneuvers, increase process noise or gate threshold.

Compare several profiles rather than optimizing only RMSE. Consistency and recovery behavior matter as much as average error.
