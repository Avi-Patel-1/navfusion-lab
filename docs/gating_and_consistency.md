# Gating and Consistency

All filter updates compute normalized innovation squared:

```text
NIS = y^T S^-1 y
```

where `y` is the residual and `S` is the innovation covariance. A measurement is accepted only if NIS is below the selected profile gate.

## Residual Wrapping

Bearing and heading residuals are wrapped to `[-pi, pi)` before gating. This prevents discontinuities near the angle boundary.

## Joseph Update

Accepted linearized updates use:

```text
P = (I - K H) P (I - K H)^T + K R K^T
```

The matrix is then symmetrized and repaired with a small positive floor.

## Dropout Handling

GPS dropout samples are retained in `measurements.csv` with `valid = 0`. They are not used for updates. The predictor can inflate covariance during dropout windows so the filter remains receptive when GPS returns.
