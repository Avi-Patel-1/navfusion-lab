# MATLAB and Simulink Notes

The `matlab/` folder contains plain MATLAB functions for the prediction and measurement-update equations. The `simulink/README.md` file describes a block-level reconstruction.

## Reference Export

```bash
python3 -m fusion_sandbox export-matlab --config examples/configs/ekf_imu_gps_baro.json --out matlab/reference
```

The export writes truth, measurements, the resolved config, and a manifest. These files are suitable for MATLAB timetables or From Workspace blocks.

## Matching a Run

Use the same tuning profile, gate threshold, initial estimate offset, sample times, and dropout windows. Compare RMSE, accepted/rejected counts, and final covariance trace against the Python summary.
