from __future__ import annotations

from typing import Any

import numpy as np

from .config import normalize_config, validate_config
from .sensors import BarometerSensor, DopplerVelocitySensor, GNSSPseudorangeSensor, GPSSensor, IMUSensor, MagnetometerSensor, RadarAltimeterSensor, RangeBeaconSensor, WheelOdometrySensor
from .trajectory import generate_truth
from .trajectory.profiles import acceleration_for_profile


SENSOR_ORDER = {
    "imu": 0,
    "gps": 1,
    "barometer": 2,
    "magnetometer": 3,
    "range_beacon": 4,
    "wheel_odometry": 5,
    "radar_altimeter": 6,
    "doppler_velocity": 7,
    "gnss_pseudorange": 8,
}


def true_accel(t: float) -> np.ndarray:
    return acceleration_for_profile("mixed", float(t), np.zeros(3), np.zeros(3), 60.0)


def simulate_navigation(config: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    cfg = normalize_config(config)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Invalid config:\n" + "\n".join(errors))
    truth = generate_truth(cfg)
    rng = np.random.default_rng(int(cfg.get("seed", 314)))
    sensor_cfg = cfg["sensors"]
    sensors = [
        IMUSensor(sensor_cfg["imu"]),
        GPSSensor(sensor_cfg["gps"]),
        BarometerSensor(sensor_cfg["barometer"]),
        MagnetometerSensor(sensor_cfg["magnetometer"]),
        RangeBeaconSensor(sensor_cfg["range_beacon"]),
        WheelOdometrySensor(sensor_cfg["wheel_odometry"]),
        RadarAltimeterSensor(sensor_cfg["radar_altimeter"]),
        DopplerVelocitySensor(sensor_cfg["doppler_velocity"]),
        GNSSPseudorangeSensor(sensor_cfg["gnss_pseudorange"]),
    ]
    measurements: list[dict[str, float]] = []
    for sensor in sensors:
        measurements.extend(sensor.generate(truth, rng))
    measurements.sort(key=lambda row: (float(row["time_s"]), SENSOR_ORDER.get(str(row["sensor"]), 99), float(row.get("beacon_id", -1.0))))
    return truth, measurements


def simulate_measurements(config: dict[str, Any]) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    """Compatibility wrapper returning truth and a time-wide measurement table."""
    truth, events = simulate_navigation(config)
    by_time: dict[float, list[dict[str, float]]] = {}
    for event in events:
        by_time.setdefault(float(event["measurement_time_s"]), []).append(event)
    rows: list[dict[str, float]] = []
    for tr in truth:
        t = float(tr["time_s"])
        row = {
            "time_s": t,
            "imu_ax": float("nan"),
            "imu_ay": float("nan"),
            "imu_az": float("nan"),
            "gps_x": float("nan"),
            "gps_y": float("nan"),
            "gps_z": float("nan"),
            "gps_vx": float("nan"),
            "gps_vy": float("nan"),
            "gps_vz": float("nan"),
            "gps_valid": 0.0,
            "baro_z": float("nan"),
            "baro_valid": 0.0,
        }
        for event in by_time.get(t, []):
            sensor = event["sensor"]
            if sensor == "imu":
                row.update({"imu_ax": event["ax"], "imu_ay": event["ay"], "imu_az": event["az"]})
            elif sensor == "gps":
                row.update(
                    {
                        "gps_x": event["x"],
                        "gps_y": event["y"],
                        "gps_z": event["z"],
                        "gps_vx": event["vx"],
                        "gps_vy": event["vy"],
                        "gps_vz": event["vz"],
                        "gps_valid": event["valid"],
                    }
                )
            elif sensor == "barometer":
                row.update({"baro_z": event["altitude_m"], "baro_valid": event["valid"]})
        rows.append(row)
    return truth, rows
