from __future__ import annotations

from typing import Iterable


def group_by_sensor(events: Iterable[dict[str, float]]) -> dict[str, list[dict[str, float]]]:
    grouped: dict[str, list[dict[str, float]]] = {}
    for event in events:
        grouped.setdefault(str(event.get("sensor", "unknown")), []).append(event)
    return grouped
