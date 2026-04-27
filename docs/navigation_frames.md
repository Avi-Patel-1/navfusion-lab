# Navigation Frames

The `navigation/` package adds real navigation-frame utilities while keeping the simulation demos lightweight.

## WGS84

`navigation.wgs84` defines WGS84 semi-major axis, flattening, eccentricity, Earth rotation rate, and normal gravity. It provides:

- `geodetic_to_ecef(lat, lon, alt)`
- `ecef_to_geodetic(ecef)`
- `ecef_to_enu(ecef, reference_lla)`
- `enu_to_ecef(enu, reference_lla)`
- `ecef_to_ned(ecef, reference_lla)`
- `ned_to_ecef(ned, reference_lla)`
- `gravity_normal(lat, alt)`

Angles are radians and distances are meters.

## Strapdown INS

`navigation.ins` provides quaternion propagation and a local-frame strapdown mechanization step:

```text
corrected gyro -> quaternion update
corrected accelerometer -> navigation acceleration
position and velocity integration
```

The `StrapdownState` includes position, velocity, attitude quaternion, gyro bias, and accelerometer bias.

## Coning, Sculling, and Preintegration

Coning and sculling corrections are available as small vector utilities. `IMUPreintegrator` accumulates delta angle, delta velocity, delta position, elapsed time, and sample count for delayed-update or batch-estimation studies.

## Attitude Filters

The package includes quaternion attitude filters for lower-level inertial studies:

- `ComplementaryAttitudeFilter` combines gyro prediction with accelerometer and optional magnetometer vector correction.
- `MahonyAttitudeFilter` adds proportional/integral vector-error feedback and estimates gyro bias.
- `MadgwickAttitudeFilter` applies a gradient-descent accelerometer correction to gyro propagation.

These filters expose `update(...)` and return roll, pitch, yaw, and quaternion estimates.

## Validation

Frame round-trip tests cover geodetic/ECEF and local ENU/NED conversions. Mechanization and attitude tests verify stationary acceleration balance, quaternion normalization, gyro yaw integration, and tilt correction.
