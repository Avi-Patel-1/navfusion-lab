from __future__ import annotations

import numpy as np

from ..math_utils import nearest_positive_definite, safe_inverse


def rts_smoother(
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    transition_matrices: np.ndarray,
    process_noises: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Run a Rauch-Tung-Striebel backward pass for linearized filter histories."""
    x_f = np.asarray(filtered_states, dtype=float)
    P_f = np.asarray(filtered_covariances, dtype=float)
    F = np.asarray(transition_matrices, dtype=float)
    Q = np.asarray(process_noises, dtype=float)
    if x_f.ndim != 2:
        raise ValueError("filtered_states must have shape (steps, state_dim)")
    steps, state_dim = x_f.shape
    if P_f.shape != (steps, state_dim, state_dim):
        raise ValueError("filtered_covariances must have shape (steps, state_dim, state_dim)")
    if F.shape != (max(steps - 1, 0), state_dim, state_dim):
        raise ValueError("transition_matrices must have shape (steps - 1, state_dim, state_dim)")
    if Q.shape != (max(steps - 1, 0), state_dim, state_dim):
        raise ValueError("process_noises must have shape (steps - 1, state_dim, state_dim)")
    x_s = x_f.copy()
    P_s = P_f.copy()
    for k in range(steps - 2, -1, -1):
        x_pred = F[k] @ x_f[k]
        P_pred = nearest_positive_definite(F[k] @ P_f[k] @ F[k].T + Q[k])
        gain = P_f[k] @ F[k].T @ safe_inverse(P_pred)
        x_s[k] = x_f[k] + gain @ (x_s[k + 1] - x_pred)
        P_s[k] = nearest_positive_definite(P_f[k] + gain @ (P_s[k + 1] - P_pred) @ gain.T)
    return x_s, P_s


def fixed_lag_smoother(
    filtered_states: np.ndarray,
    filtered_covariances: np.ndarray,
    transition_matrices: np.ndarray,
    process_noises: np.ndarray,
    lag: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply RTS smoothing over sliding windows and release fixed-lag estimates."""
    x_f = np.asarray(filtered_states, dtype=float)
    P_f = np.asarray(filtered_covariances, dtype=float)
    F = np.asarray(transition_matrices, dtype=float)
    Q = np.asarray(process_noises, dtype=float)
    if int(lag) <= 0 or x_f.shape[0] <= 1:
        return x_f.copy(), P_f.copy()
    steps = x_f.shape[0]
    smoothed_x = x_f.copy()
    smoothed_P = P_f.copy()
    window = int(lag) + 1
    for end in range(1, steps):
        start = max(0, end - window + 1)
        xs, Ps = rts_smoother(x_f[start : end + 1], P_f[start : end + 1], F[start:end], Q[start:end])
        release_index = start
        smoothed_x[release_index] = xs[0]
        smoothed_P[release_index] = Ps[0]
    tail_start = max(0, steps - window)
    xs, Ps = rts_smoother(x_f[tail_start:], P_f[tail_start:], F[tail_start : steps - 1], Q[tail_start : steps - 1])
    smoothed_x[tail_start:] = xs
    smoothed_P[tail_start:] = Ps
    return smoothed_x, smoothed_P


class FixedLagSmoother:
    """Streaming fixed-lag RTS smoother for linearized histories."""

    def __init__(self, lag: int):
        if int(lag) < 0:
            raise ValueError("lag must be nonnegative")
        self.lag = int(lag)
        self.states: list[np.ndarray] = []
        self.covariances: list[np.ndarray] = []
        self.transitions: list[np.ndarray] = []
        self.process_noises: list[np.ndarray] = []

    def add(
        self,
        filtered_state: np.ndarray,
        filtered_covariance: np.ndarray,
        transition_from_previous: np.ndarray | None = None,
        process_noise_from_previous: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray] | None:
        state = np.asarray(filtered_state, dtype=float).reshape(-1)
        covariance = nearest_positive_definite(np.asarray(filtered_covariance, dtype=float))
        if self.states:
            if transition_from_previous is None or process_noise_from_previous is None:
                raise ValueError("transition and process noise are required after the first state")
            self.transitions.append(np.asarray(transition_from_previous, dtype=float))
            self.process_noises.append(np.asarray(process_noise_from_previous, dtype=float))
        self.states.append(state)
        self.covariances.append(covariance)
        if len(self.states) <= self.lag + 1:
            return None
        released = self._smooth_buffer()[0]
        self.states.pop(0)
        self.covariances.pop(0)
        if self.transitions:
            self.transitions.pop(0)
        if self.process_noises:
            self.process_noises.pop(0)
        return released

    def _smooth_buffer(self) -> list[tuple[np.ndarray, np.ndarray]]:
        if len(self.states) == 1:
            return [(self.states[0].copy(), self.covariances[0].copy())]
        xs, Ps = rts_smoother(np.asarray(self.states), np.asarray(self.covariances), np.asarray(self.transitions), np.asarray(self.process_noises))
        return [(xs[i].copy(), Ps[i].copy()) for i in range(xs.shape[0])]

    def flush(self) -> list[tuple[np.ndarray, np.ndarray]]:
        result = self._smooth_buffer()
        self.states.clear()
        self.covariances.clear()
        self.transitions.clear()
        self.process_noises.clear()
        return result
