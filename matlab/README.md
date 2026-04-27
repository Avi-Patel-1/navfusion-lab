# MATLAB Reference Notes

The MATLAB folder mirrors the core estimator equations used by the Python sandbox. The scripts are intentionally plain MATLAB functions so the equations can be copied into MATLAB Function blocks or used for reference checks.

## Files

- `ekf_predict.m`: inertial time update for the 9-state navigation model.
- `ekf_update_gps.m`: Cartesian GPS position update.
- `ekf_update_gps_range_bearing.m`: nonlinear range, bearing, altitude GPS update.
- `ekf_update_baro.m`: scalar barometer altitude update.
- `ukf_predict_update.m`: compact unscented predict/update reference.
- `generate_reference_data.m`: exports Python truth and measurement CSV files into `matlab/reference`.
- `run_demo.m`: runs the baseline experiment and refreshes MATLAB reference data.

## Validation Flow

1. Run `run_demo` from MATLAB.
2. Inspect `outputs/matlab_demo/summary.json` for baseline metrics.
3. Load `matlab/reference/truth_reference.csv` and `measurement_reference.csv`.
4. Compare MATLAB states against the Python `estimates_<filter>_<profile>.csv` files.
5. Confirm covariance matrices remain symmetric after each predict and update.
