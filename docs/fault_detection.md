# Fault Detection

The fault-detection layer turns innovation logs into sensor health products. It is separate from estimator gating so post-run analysis can be rebuilt without rerunning a scenario.

## Innovation Gating

Estimator updates already write NIS, gate value, acceptance flag, residual norm, source, and truth fault labels when available. These fields are the inputs for health scoring.

## Sensor Health Monitor

`SensorHealthMonitor` keeps an exponentially smoothed score per sensor family. Rejections or high NIS reduce the score; accepted low-NIS samples raise it. Sensors can enter quarantine and recover after stable residuals.

## Consistency Checks

The module includes 95 percent and 99 percent chi-square threshold tables for dimensions one through six. A residual whiteness proxy uses autocorrelation energy to flag serially correlated residuals.

## CLI

```bash
python3 -m fusion_sandbox analyze-faults --input outputs/demo/innovations_ekf_nominal.csv --out outputs/demo/fault_analysis.json
```

The output includes a timeline and per-sensor summaries with detection counts, truth fault labels, confusion matrix, false alarm rate, missed detection rate, detection latency, final health score, quarantine count, and whiteness score.
