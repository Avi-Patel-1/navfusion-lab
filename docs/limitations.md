# Limitations

- The default demo pipeline estimates in a local tangent frame; WGS84 and local ENU/NED utilities are available for frame conversion and validation.
- IMU acceleration is modeled in the navigation frame for the high-level demos; a strapdown body-frame mechanization is available in `navigation.ins`.
- Heading is derived from velocity rather than estimated as a separate state.
- Sensor noise is Gaussian except for explicit dropout, latency, and outlier events.
- Plotting uses simple SVG writers to keep the project NumPy-only.

These choices keep the sandbox easy to run in a clean Python environment while still exercising practical filter tuning and validation behavior.
