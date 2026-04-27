from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ManeuverSegment:
    start_s: float
    end_s: float
    value: float

    def active(self, t: float) -> bool:
        return self.start_s <= t <= self.end_s


def segment_value(t: float, segments: list[ManeuverSegment], default: float = 0.0) -> float:
    for segment in segments:
        if segment.active(t):
            return segment.value
    return default
