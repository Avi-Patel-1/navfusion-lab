# Filter Design

## Navigation State

The primary filters use a 9-state local tangent frame model:

```text
x = [px, py, pz, vx, vy, vz, bax, bay, baz]^T
```

Position and velocity are expressed in meters and meters per second. The last three states estimate accelerometer bias in the navigation frame. Heading is derived from horizontal velocity for magnetometer updates and reporting.

## Time Update

```text
a_hat = a_imu - b_a
p_k+1 = p_k + v_k dt + 0.5 a_hat dt^2
v_k+1 = v_k + a_hat dt
b_k+1 = b_k
P_k+1 = F P_k F^T + Q
```

The process noise is driven by profile-specific acceleration noise and bias random walk. During configured GPS dropout windows, covariance can be inflated by the selected tuning profile to keep the filter from becoming overconfident.

## Linear KF

The linear filter uses Cartesian position, velocity, barometer, and wheel-odometry updates. It is useful as a baseline for constant-velocity navigation examples.

## Square-Root and Information Forms

The square-root filter keeps a Cholesky factor synchronized with covariance to exercise a numerically stronger covariance representation. The information filter keeps information matrix and vector views synchronized with the covariance state.

## EKF

The EKF shares the same time update and adds nonlinear measurements:

- GPS range, bearing, altitude from Cartesian GPS.
- Range to fixed beacons.
- Heading from magnetometer using the velocity-derived yaw model.

Each update is linearized at the current estimate and uses Joseph-form covariance correction.

## UKF

The UKF uses sigma points for prediction and nonlinear measurements. It is implemented with NumPy only and is intended for range-beacon and other nonlinear-measurement comparisons where analytic Jacobians are a source of tuning risk.

## Robust, Adaptive, Particle, and ABG Filters

The robust filter scales high residuals with a Huber-style rule before update. The adaptive filter adjusts measurement covariance from recent innovation energy. The particle filter uses bootstrap prediction and likelihood weighting for nonlinear/non-Gaussian runs. The alpha-beta-gamma tracker provides a simple deterministic baseline for smooth position-aided motion.

## Covariance Handling

Every covariance update is symmetrized and repaired with a small positive eigenvalue floor. This avoids negative diagonal entries from roundoff while keeping the numerical behavior visible in the metrics.
