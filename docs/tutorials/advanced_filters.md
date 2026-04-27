# Tutorial: Advanced Filter Comparison

Run the advanced comparison:

```bash
python3 -m fusion_sandbox compare --config examples/configs/advanced_filter_comparison.json --out outputs/advanced_filter_check
```

The config compares linear, square-root, information, EKF, robust, adaptive, and particle filters against the same nonlinear measurement stream.

For a simple deterministic baseline, run:

```bash
python3 -m fusion_sandbox run --config examples/configs/alpha_beta_gamma_baseline.json --out outputs/alpha_beta_gamma_baseline
```

Use `summary.json` to compare RMSE and update counts. Use `report.html` for plots and the manifest for a full output inventory.
