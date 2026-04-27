# Reporting

Every run writes machine-readable logs and static plots.

## Per-Filter Artifacts

- `estimates_<filter>_<profile>.csv`
- `innovations_<filter>_<profile>.csv`
- `metrics_<filter>_<profile>.json`
- `plots/position_error_<filter>_<profile>.svg`
- `plots/velocity_error_<filter>_<profile>.svg`
- `plots/altitude_<filter>_<profile>.svg`
- `plots/covariance_bounds_<filter>_<profile>.svg`
- `plots/nis_<filter>_<profile>.svg`
- `plots/residuals_<filter>_<profile>.svg`
- `plots/bias_estimates_<filter>_<profile>.svg`

## Run-Level Artifacts

- `truth.csv`
- `measurements.csv`
- `config_resolved.json`
- `summary.json`
- `manifest.json`
- `report.html`
- comparison bar charts

## Report Rebuild

Reports can be rebuilt from an existing run directory:

```bash
python3 -m fusion_sandbox report --run outputs/demo
```

This refreshes `report.html` and `manifest.json` without rerunning the filter.

## Embedded Reference Export

`export-c-header` converts numeric CSV traces into C arrays:

```bash
python3 -m fusion_sandbox export-c-header --run outputs/demo --out outputs/demo/reference_vectors.h
```

This is intended for embedded validation, hardware-in-the-loop smoke tests, and compact regression fixtures.
