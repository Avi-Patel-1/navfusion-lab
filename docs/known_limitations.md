# Known Limitations

- The default pipeline estimates in a local Cartesian frame. WGS84 transforms and strapdown INS utilities are available, but full ECEF filtering is not yet the default run mode.
- The main estimator state does not yet include attitude; heading is derived from horizontal velocity unless using lower-level navigation utilities.
- The particle filter is intended as a nonlinear/non-Gaussian study baseline and needs scenario-specific tuning for high-accuracy use.
- The alpha-beta-gamma filter is a deterministic baseline for smooth position-aided motion, not a replacement for the Kalman filters.
- Sensor models cover several practical faults, but star trackers, visual odometry, optical flow, and GNSS carrier-phase processing are not yet implemented.
- Static SVG plotting keeps dependencies small; interactive dashboards are outside the current core.

These limits are deliberate boundaries for the current implementation batch and are documented so future work can be prioritized cleanly.
