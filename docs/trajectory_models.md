# Trajectory Models

Truth is sampled in a local tangent frame at `truth_sample_rate_hz`. The simulator records position, velocity, acceleration, roll, pitch, and yaw.

## Profiles

- `straight_line`: constant velocity.
- `coordinated_turn`: lateral acceleration with smooth heading change.
- `climb_descent`: vertical maneuver sequence for barometer tests.
- `aggressive_accel`: acceleration pulses and lateral maneuvering.
- `hover_low_speed`: damped low-speed motion for weak-observability checks.
- `figure_eight`: analytic figure-eight position, velocity, and acceleration.
- `mixed`: default multi-axis maneuver profile.

## 2D and 3D Modes

In `2d` mode, vertical position, velocity, and acceleration are pinned to zero. In `3d` mode, all channels are active.

## Interpolation

Truth interpolation is available for stale or delayed measurements. The pipeline stores both availability time and original measurement time so latency effects can be measured.
