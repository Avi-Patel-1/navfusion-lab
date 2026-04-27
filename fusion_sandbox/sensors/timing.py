from __future__ import annotations


def should_sample(t: float, rate_hz: float, eps: float = 1e-9) -> bool:
    period = 1.0 / float(rate_hz)
    k = round(t / period)
    return abs(t - k * period) <= max(eps, period * 1e-6)


def available_time(measurement_time_s: float, latency_s: float) -> float:
    return round(float(measurement_time_s) + float(latency_s), 10)


def sample_times(duration_s: float, rate_hz: float, include_endpoint: bool = False) -> list[float]:
    period = 1.0 / float(rate_hz)
    duration = float(duration_s)
    count = int(duration / period) + 1
    times = [round(i * period, 10) for i in range(count + 1) if i * period <= duration + 1e-9]
    if not times or times[0] != 0.0:
        times.insert(0, 0.0)
    if include_endpoint and abs(times[-1] - duration) > 1e-9:
        times.append(round(duration, 10))
    return times
