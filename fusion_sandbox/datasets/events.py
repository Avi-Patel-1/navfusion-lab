from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


REQUIRED_EVENT_FIELDS = {"time_s", "sensor", "valid"}


def _coerce(value: str) -> Any:
    if value == "":
        return value
    try:
        return float(value)
    except ValueError:
        return value


def read_event_csv(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(newline="") as f:
        return [{key: _coerce(value) for key, value in row.items()} for row in csv.DictReader(f)]


def write_jsonl(path: str | Path, events: list[dict[str, Any]]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for event in events:
            f.write(json.dumps(event, sort_keys=True) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                event = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSONL at line {line_no}: {exc}") from exc
            if not isinstance(event, dict):
                raise ValueError(f"JSONL line {line_no} must contain an object")
            events.append(event)
    return events


def validate_events(events: list[dict[str, Any]]) -> list[str]:
    errors: list[str] = []
    last_time = -float("inf")
    for i, event in enumerate(events):
        missing = REQUIRED_EVENT_FIELDS - set(event)
        if missing:
            errors.append(f"event[{i}] missing fields: {sorted(missing)}")
        try:
            t = float(event.get("time_s", 0.0))
        except (TypeError, ValueError):
            errors.append(f"event[{i}].time_s must be numeric")
            continue
        if t < last_time:
            errors.append(f"event[{i}] time_s is not monotonic")
        last_time = t
        if "measurement_time_s" in event and float(event["measurement_time_s"]) > t + 3600.0:
            errors.append(f"event[{i}].measurement_time_s is implausibly later than availability time")
        if "valid" in event and float(event["valid"]) not in {0.0, 1.0}:
            errors.append(f"event[{i}].valid must be 0 or 1")
    return errors


def summarize_events(events: list[dict[str, Any]]) -> dict[str, Any]:
    by_sensor: dict[str, dict[str, Any]] = {}
    for event in events:
        sensor = str(event.get("sensor", "unknown"))
        bucket = by_sensor.setdefault(sensor, {"count": 0, "valid": 0, "invalid": 0, "outliers": 0, "dropouts": 0})
        bucket["count"] += 1
        bucket["valid"] += int(float(event.get("valid", 1.0)) >= 0.5)
        bucket["invalid"] += int(float(event.get("valid", 1.0)) < 0.5)
        bucket["outliers"] += int(float(event.get("is_outlier", 0.0)) >= 0.5)
        bucket["dropouts"] += int(float(event.get("is_dropout", 0.0)) >= 0.5)
    times = [float(event["time_s"]) for event in events if "time_s" in event]
    return {
        "event_count": len(events),
        "sensor_count": len(by_sensor),
        "time_start_s": min(times) if times else 0.0,
        "time_end_s": max(times) if times else 0.0,
        "by_sensor": by_sensor,
    }
