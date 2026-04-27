from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..math_utils import interpolate_records
from .profiles import acceleration_for_profile, figure_eight_state


def _record(t: float, pos: np.ndarray, vel: np.ndarray, acc: np.ndarray, previous_yaw: float) -> dict[str, float]:
    horizontal_speed = float(np.linalg.norm(vel[:2]))
    yaw = math.atan2(float(vel[1]), float(vel[0])) if horizontal_speed > 1e-6 else previous_yaw
    pitch = math.atan2(float(vel[2]), max(horizontal_speed, 1e-6))
    roll = math.atan2(float(acc[1]), 9.80665)
    return {
        "time_s": round(float(t), 10),
        "px": float(pos[0]),
        "py": float(pos[1]),
        "pz": float(pos[2]),
        "vx": float(vel[0]),
        "vy": float(vel[1]),
        "vz": float(vel[2]),
        "ax": float(acc[0]),
        "ay": float(acc[1]),
        "az": float(acc[2]),
        "roll_rad": float(roll),
        "pitch_rad": float(pitch),
        "yaw_rad": float(yaw),
    }


def generate_truth(config: dict[str, Any]) -> list[dict[str, float]]:
    traj = config["trajectory"]
    dt = 1.0 / float(config["truth_sample_rate_hz"])
    duration = float(config["duration_s"])
    steps = int(round(duration / dt)) + 1
    profile = str(traj["profile"])
    mode = str(traj.get("mode", "3d"))
    pos = np.array(traj["initial_position_m"], dtype=float)
    vel = np.array(traj["initial_velocity_mps"], dtype=float)
    if mode == "2d":
        pos[2] = 0.0
        vel[2] = 0.0
    records: list[dict[str, float]] = []
    previous_yaw = math.atan2(float(vel[1]), float(vel[0])) if np.linalg.norm(vel[:2]) > 1e-6 else 0.0
    if profile == "figure_eight":
        radius = float(traj.get("figure_eight_radius_m", 90.0))
        for i in range(steps):
            t = i * dt
            p, v, a = figure_eight_state(t, duration, pos, radius)
            if mode == "2d":
                p[2] = 0.0
                v[2] = 0.0
                a[2] = 0.0
            rec = _record(t, p, v, a, previous_yaw)
            previous_yaw = rec["yaw_rad"]
            records.append(rec)
        return records
    for i in range(steps):
        t = i * dt
        acc = acceleration_for_profile(profile, t, pos, vel, duration)
        if mode == "2d":
            acc[2] = 0.0
        rec = _record(t, pos.copy(), vel.copy(), acc.copy(), previous_yaw)
        previous_yaw = rec["yaw_rad"]
        records.append(rec)
        pos = pos + vel * dt + 0.5 * acc * dt * dt
        vel = vel + acc * dt
        if mode == "2d":
            pos[2] = 0.0
            vel[2] = 0.0
    return records


def truth_at(truth: list[dict[str, float]], t: float) -> dict[str, float]:
    fields = ["px", "py", "pz", "vx", "vy", "vz", "ax", "ay", "az", "yaw_rad"]
    values = interpolate_records(truth, t, fields)
    return {"time_s": float(t), **{field: float(value) for field, value in zip(fields, values)}}
