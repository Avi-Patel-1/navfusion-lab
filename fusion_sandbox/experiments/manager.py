from __future__ import annotations

import copy
import hashlib
import itertools
import json
from typing import Any

from ..config import deep_merge, normalize_config, validate_config


def config_hash(config: dict[str, Any]) -> str:
    resolved = normalize_config(config)
    payload = json.dumps(resolved, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()[:16]


def _set_deep(base: dict[str, Any], dotted_path: str, value: Any) -> dict[str, Any]:
    result = copy.deepcopy(base)
    cursor = result
    parts = dotted_path.split(".")
    for part in parts[:-1]:
        cursor = cursor.setdefault(part, {})
    cursor[parts[-1]] = value
    return result


def expand_scenario_matrix(base_config: dict[str, Any], matrix: dict[str, list[Any]]) -> list[dict[str, Any]]:
    keys = sorted(matrix)
    scenarios = []
    for values in itertools.product(*(matrix[key] for key in keys)):
        overrides: dict[str, Any] = {}
        label_parts = []
        for key, value in zip(keys, values):
            overrides = _set_deep(overrides, key, value)
            label_parts.append(f"{key.split('.')[-1]}={value}")
        scenario = deep_merge(base_config, overrides)
        scenario["name"] = "_".join(str(part).replace("/", "-") for part in label_parts)
        scenario["experiments"] = []
        scenarios.append(normalize_config(scenario))
    return scenarios


def plan_experiment(config: dict[str, Any], matrix: dict[str, list[Any]] | None = None) -> dict[str, Any]:
    base = normalize_config(config)
    if matrix:
        scenarios = expand_scenario_matrix(base, matrix)
    else:
        scenarios = []
        experiments = base.get("experiments", [])
        for experiment in experiments or [{"name": base.get("name", "scenario"), "overrides": {}}]:
            scenario = deep_merge(base, experiment.get("overrides", {}))
            scenario["name"] = experiment.get("name", scenario.get("name", "scenario"))
            scenario["experiments"] = []
            scenarios.append(normalize_config(scenario))
    rows = []
    for scenario in scenarios:
        errors = validate_config(scenario)
        rows.append(
            {
                "name": scenario.get("name", "scenario"),
                "hash": config_hash(scenario),
                "valid": not errors,
                "errors": errors,
                "filters": scenario["estimator"]["filters"],
                "profiles": scenario["estimator"]["profiles"],
                "duration_s": scenario["duration_s"],
            }
        )
    return {"scenario_count": len(rows), "scenarios": rows}
