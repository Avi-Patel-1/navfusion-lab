from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ..calibration.noise import residual_autocorrelation


CHI_SQUARE_95 = {1: 3.841, 2: 5.991, 3: 7.815, 4: 9.488, 5: 11.070, 6: 12.592}
CHI_SQUARE_99 = {1: 6.635, 2: 9.210, 3: 11.345, 4: 13.277, 5: 15.086, 6: 16.812}


def whiteness_p_value_proxy(residuals: np.ndarray, max_lag: int = 10) -> float:
    ac = np.asarray(residual_autocorrelation(residuals, max_lag=max_lag)[1:], dtype=float)
    if ac.size == 0:
        return 1.0
    score = float(np.sum(ac * ac) * max(len(residuals), 1))
    return float(np.exp(-0.5 * score))


@dataclass
class SensorHealthMonitor:
    quarantine_threshold: float = 0.25
    recovery_threshold: float = 0.72
    recovery_count: int = 5
    scores: dict[str, float] = field(default_factory=dict)
    stable_counts: dict[str, int] = field(default_factory=dict)
    quarantined: dict[str, bool] = field(default_factory=dict)

    def update(self, source: str, nis: float, dim: int, accepted: bool, time_s: float) -> dict[str, Any]:
        threshold = CHI_SQUARE_99.get(max(1, min(int(dim), 6)), 16.812)
        source_key = source.split("_")[0]
        score = self.scores.get(source_key, 1.0)
        fault_like = (not accepted) or float(nis) > threshold
        score = 0.85 * score + 0.15 * (0.0 if fault_like else 1.0)
        self.scores[source_key] = score
        if fault_like:
            self.stable_counts[source_key] = 0
        else:
            self.stable_counts[source_key] = self.stable_counts.get(source_key, 0) + 1
        was_quarantined = self.quarantined.get(source_key, False)
        if score < self.quarantine_threshold:
            self.quarantined[source_key] = True
            reason = "quarantine"
        elif was_quarantined and score > self.recovery_threshold and self.stable_counts[source_key] >= self.recovery_count:
            self.quarantined[source_key] = False
            reason = "recover"
        else:
            reason = "fault_like" if fault_like else "nominal"
        return {
            "time_s": float(time_s),
            "sensor": source_key,
            "source": source,
            "nis": float(nis),
            "health_score": float(score),
            "quarantined": bool(self.quarantined.get(source_key, False)),
            "reason": reason,
        }


def analyze_faults(innovations: list[dict[str, Any]]) -> dict[str, Any]:
    monitor = SensorHealthMonitor()
    timeline = []
    residuals_by_sensor: dict[str, list[float]] = {}
    for row in innovations:
        source = str(row.get("source", "unknown"))
        residual_cols = [key for key in row if key.startswith("residual_")]
        dim = max(len(residual_cols), 1)
        event = monitor.update(source, float(row.get("nis", 0.0)), dim, float(row.get("accepted", 1.0)) >= 0.5, float(row.get("time_s", 0.0)))
        event["truth_outlier_label"] = float(row.get("sensor_is_outlier", 0.0)) >= 0.5
        event["truth_dropout_label"] = float(row.get("sensor_is_dropout", 0.0)) >= 0.5
        timeline.append(event)
        residuals_by_sensor.setdefault(event["sensor"], []).append(float(row.get("residual_norm", 0.0)))
    sensor_summary = {}
    for sensor, values in residuals_by_sensor.items():
        sensor_events = [event for event in timeline if event["sensor"] == sensor]
        quarantined_count = sum(1 for event in sensor_events if event["quarantined"])
        truth_faults = sum(1 for event in sensor_events if (event["truth_outlier_label"] or event["truth_dropout_label"]))
        detections = sum(1 for event in sensor_events if event["reason"] in {"quarantine", "fault_like"} or event["quarantined"])
        tp = fp = tn = fn = 0
        first_truth_time: float | None = None
        first_detection_after_truth: float | None = None
        for event in sensor_events:
            truth_fault = bool(event["truth_outlier_label"] or event["truth_dropout_label"])
            detected = bool(event["reason"] in {"quarantine", "fault_like"} or event["quarantined"])
            if truth_fault and first_truth_time is None:
                first_truth_time = float(event["time_s"])
            if first_truth_time is not None and detected and first_detection_after_truth is None and float(event["time_s"]) >= first_truth_time:
                first_detection_after_truth = float(event["time_s"])
            if truth_fault and detected:
                tp += 1
            elif truth_fault and not detected:
                fn += 1
            elif not truth_fault and detected:
                fp += 1
            else:
                tn += 1
        detection_latency = None if first_truth_time is None or first_detection_after_truth is None else first_detection_after_truth - first_truth_time
        sensor_summary[sensor] = {
            "samples": len(values),
            "quarantined_count": quarantined_count,
            "detections": detections,
            "truth_fault_labels": truth_faults,
            "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
            "false_alarm_rate": float(fp / max(fp + tn, 1)),
            "missed_detection_rate": float(fn / max(fn + tp, 1)),
            "detection_latency_s": detection_latency,
            "whiteness_p_proxy": whiteness_p_value_proxy(np.asarray(values, dtype=float)),
            "final_health_score": monitor.scores.get(sensor, 1.0),
        }
    return {"timeline": timeline, "sensor_summary": sensor_summary}
