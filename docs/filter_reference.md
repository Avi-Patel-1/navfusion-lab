# Filter Reference

All filters expose `predict(...)` plus sensor-specific update methods. The common state vector is:

```text
[px, py, pz, vx, vy, vz, bax, bay, baz]
```

The alpha-beta-gamma baseline reuses the final three slots as acceleration estimates; metrics still report them through the common trace format.

## Linear KF

The linear filter uses the shared constant-acceleration time update and linear GPS, barometer, and wheel updates. It is the reference for simple Cartesian scenarios.

## Square-Root KF

`SquareRootKalmanFilter` keeps a Cholesky factor synchronized with covariance after prediction and update. This gives the project a numerically oriented baseline without changing the public API.

## Information Filter

`InformationKalmanFilter` maintains information matrix and vector views of the covariance form. It is useful for comparing sparse-measurement behavior and for future factor-style extensions.

## EKF and UKF

The EKF handles nonlinear range beacon and magnetometer heading updates with analytic linearization. The UKF uses sigma points for nonlinear transformations and avoids hand-derived Jacobians where appropriate.

## Robust and Adaptive Filters

The robust filter applies Huber-style residual scaling before the Kalman update. The adaptive filter adjusts measurement covariance from recent innovation energy, which helps show the tradeoff between responsiveness and consistency.

## Particle Filter

The particle filter uses bootstrap prediction and likelihood updates for GPS, barometer, and range beacons. It reports effective sample size in innovation rows.

## Alpha-Beta-Gamma Baseline

The alpha-beta-gamma baseline is a deterministic tracker for position-aided motion. It is intentionally simple and best suited to smooth trajectories with frequent position updates.

## Attitude Filters

Navigation attitude utilities include complementary, Mahony-style, and Madgwick-style quaternion filters. They are available under `fusion_sandbox.navigation` and are intended for attitude propagation and lower-level inertial studies alongside the position/velocity estimators.

## Smoothing

The RTS smoother in `filters/smoothing.py` performs backward covariance-aware smoothing for fixed transition and process-noise arrays. It is currently exposed as a library helper and covered by tests.

`fixed_lag_smoother` applies RTS passes over sliding windows and returns lagged estimates for online-style smoothing. `FixedLagSmoother` provides the same behavior as a streaming buffer that releases states once enough future samples have arrived.
