# Metrics

Each run writes `metrics_<filter>_<profile>.json` and rolls those values into `summary.json`.

## Core Metrics

- `position_rmse_m`
- `velocity_rmse_mps`
- `bias_rmse_mps2`
- `final_position_error_m`
- `final_velocity_error_mps`
- `max_position_error_m`
- `max_error_during_dropout_m`
- `dropout_recovery_time_s`

## Consistency Metrics

- `nis_mean`
- `nis_p95`
- `mean_nees_position`
- `mean_covariance_trace`
- `final_covariance_trace`

The NEES value is a position-only proxy because the truth bias is sensor-model dependent and not always available for every source.

## Decision Counts

- `accepted_updates`
- `rejected_updates`
- `invalid_measurements`
- `updates_by_sensor`

Rejected updates are caused by gating, staleness, or unsupported sensor/filter combinations.
