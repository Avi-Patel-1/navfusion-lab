from __future__ import annotations


CONFIG_SCHEMA = {
    "name": "string experiment name",
    "seed": "integer random seed",
    "duration_s": "positive float",
    "truth_sample_rate_hz": "positive float",
    "trajectory": {
        "profile": "straight_line, coordinated_turn, climb_descent, aggressive_accel, hover_low_speed, figure_eight, mixed",
        "mode": "2d or 3d",
        "initial_position_m": "length-3 float list",
        "initial_velocity_mps": "length-3 float list",
    },
    "sensors": {
        "imu": "accelerometer and gyro noise, bias, scale, random walk, rate",
        "gps": "position/velocity rate, noise, latency, dropout windows, outlier bursts",
        "barometer": "altitude rate, noise, bias, random walk",
        "magnetometer": "heading rate, noise, disturbance events",
        "range_beacon": "nonlinear ranges to fixed beacons",
        "wheel_odometry": "horizontal velocity source",
        "radar_altimeter": "terrain-relative altitude with packet loss, jitter, warmup, and outlier support",
        "doppler_velocity": "3D velocity source with scale factor, misalignment, packet loss, jitter, and outlier support",
        "gnss_pseudorange": "raw pseudorange ranges to configured satellite positions",
    },
    "estimator": {
        "update_rate_hz": "positive float",
        "filters": "list containing linear, ekf, ukf, sqrt, information, robust, adaptive, abg, particle",
        "profiles": "list of tuning profile names",
        "max_stale_s": "nonnegative float",
    },
    "experiments": "optional list of override dictionaries for compare runs",
}


ALLOWED_TRAJECTORIES = {
    "straight_line",
    "coordinated_turn",
    "climb_descent",
    "aggressive_accel",
    "hover_low_speed",
    "figure_eight",
    "mixed",
}


ALLOWED_FILTERS = {"linear", "ekf", "ukf", "sqrt", "information", "robust", "adaptive", "abg", "particle"}


ALLOWED_PROFILES = {"aggressive", "nominal", "conservative", "dropout_robust", "high_bias"}
