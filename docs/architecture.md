# Architecture

The sandbox is organized around a reproducible simulation pipeline:

1. `config.py` normalizes JSON into a complete run definition.
2. `trajectory/` creates truth samples in a local Cartesian frame.
3. `sensors/` converts truth into asynchronous event dictionaries.
4. `filters/` predicts at the estimator rate and consumes ready events.
5. `analysis/`, `fdir/`, and `calibration/` compute performance, health, and tuning products.
6. `reports/` writes CSV, JSON, SVG, and static HTML outputs.

The core dependency remains NumPy. File IO uses the standard library so demos run in a clean Python 3.12 environment.

## Data Flow

```text
JSON config
  -> normalized config
  -> truth samples
  -> measurement events
  -> filter traces and innovation logs
  -> metrics, health analysis, plots, and handoff files
```

Each filter receives the same truth and measurement stream. That makes comparisons deterministic and keeps tuning studies fair.

## Module Boundaries

- `navigation/` contains frame transforms, strapdown propagation, and IMU preintegration.
- `filters/` owns estimator state, prediction, measurement updates, covariance handling, and smoothing helpers.
- `datasets/` handles CSV and JSONL event import/export plus schema validation.
- `fdir/` converts innovation logs into health timelines and quarantine decisions.
- `experiments/` builds reproducibility hashes and scenario plans.

## Compatibility

Existing commands and example configs remain valid. New filter names are additive, and old configs continue to normalize through the same defaults.
