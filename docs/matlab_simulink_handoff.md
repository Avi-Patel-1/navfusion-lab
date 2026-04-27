# MATLAB and Simulink Handoff

The handoff flow exports deterministic reference data and keeps MATLAB reference implementations beside the Python source.

## Reference Export

```bash
python3 -m fusion_sandbox export-matlab --config examples/configs/ekf_imu_gps_baro.json --out matlab/reference
```

The command writes truth, measurement, resolved config, and a manifest suitable for MATLAB-side verification.

## MATLAB Files

The `matlab/` directory includes prediction and update references for EKF and UKF workflows plus a demo runner. These files are intentionally readable and mirror the Python equations.

## Simulink Notes

`simulink/README.md` describes bus fields, sample times, suggested scopes, validation steps, and reconstruction checks. Use the Python CSV outputs as reference vectors when building or validating a Simulink model.

## Interface Contract

The estimator interface is:

```text
state x, covariance P, IMU sample, optional measurement event -> updated x, P, innovation record
```

This maps directly to C-like or block-diagram implementations.

## C Header Reference Vectors

Run directories can be exported as C-compatible static arrays:

```bash
python3 -m fusion_sandbox export-c-header --run outputs/demo --out outputs/demo/reference_vectors.h
```

The header includes numeric columns from truth, measurement, estimate, and innovation CSV files, plus row and column counts for each array. Use `--max-rows` for small embedded smoke-test fixtures.
