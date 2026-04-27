# Calibration

Calibration utilities work from logs so they can be used on simulated or replayed datasets.

## Residual Noise Estimation

`estimate_noise_from_innovations` reads numbered residual columns such as `residual_0` and `residual_1`. It returns mean, standard deviation, covariance, and count. The parser intentionally ignores aggregate fields such as `residual_norm`.

CLI:

```bash
python3 -m fusion_sandbox analyze-residuals --input outputs/demo/innovations_ekf_nominal.csv --out outputs/demo/residual_analysis.json
```

## Allan Deviation

`allan_deviation` computes cluster-mean Allan deviation for numeric sequences. The CLI can analyze a column from any measurement CSV:

```bash
python3 -m fusion_sandbox allan-variance --input outputs/demo/measurements.csv --column ax --sample-rate-hz 20 --out outputs/demo/allan_ax.json
```

## Tuning Use

Residual standard deviation, autocorrelation, and Allan deviation can be used to adjust process noise, measurement noise, and bias-walk terms before running `sweep-tuning`.
