# Experiment Manager

The experiment manager helps keep multi-run studies reproducible.

## Scenario Plans

`plan-experiment` resolves a config, expands its `experiments` list, validates every scenario, and assigns a deterministic hash to each resolved scenario.

```bash
python3 -m fusion_sandbox plan-experiment --config examples/configs/multi_experiment.json --out outputs/plan.json
```

## Scenario Matrix API

Library code can call `expand_scenario_matrix(base_config, matrix)` with dotted config paths:

```python
matrix = {
    "sensors.gps.position_noise_std_m": [1.5, 2.5, 4.0],
    "estimator.profiles": [["nominal"], ["dropout_robust"]],
}
```

The result is a list of normalized configs with stable names and no nested experiment lists.

## Reproducibility Hashes

Hashes are based on sorted JSON of the normalized config. Two equivalent configs get the same hash even if their input key order differs.
