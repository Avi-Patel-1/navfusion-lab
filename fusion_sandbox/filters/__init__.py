from __future__ import annotations

import numpy as np

from .base import PROFILE_MAP, PROFILES, BaseKalmanFilter, FilterProfile
from .alpha_beta_gamma import AlphaBetaGammaFilter
from .ekf import ExtendedKalmanFilter
from .information_filter import InformationKalmanFilter
from .linear_kf import LinearKalmanFilter
from .particle_filter import ParticleNavigationFilter
from .robust_kf import InnovationAdaptiveKalmanFilter, RobustHuberKalmanFilter
from .square_root_kf import SquareRootKalmanFilter
from .ukf import UnscentedKalmanFilter


Profile = FilterProfile


class NavFilter(ExtendedKalmanFilter):
    """Compatibility alias for the default navigation EKF."""

    def __init__(self, profile: FilterProfile, initial_position: np.ndarray, initial_velocity: np.ndarray | None = None):
        super().__init__(profile, initial_position, initial_velocity)


def get_profile(name: str | FilterProfile) -> FilterProfile:
    if isinstance(name, FilterProfile):
        return name
    try:
        return PROFILE_MAP[name]
    except KeyError as exc:
        raise ValueError(f"unknown tuning profile {name!r}") from exc


def create_filter(filter_type: str, profile: str | FilterProfile, initial_position: np.ndarray, initial_velocity: np.ndarray | None = None) -> BaseKalmanFilter:
    profile_obj = get_profile(profile)
    if filter_type == "linear":
        return LinearKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "sqrt":
        return SquareRootKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "information":
        return InformationKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "robust":
        return RobustHuberKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "adaptive":
        return InnovationAdaptiveKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "abg":
        return AlphaBetaGammaFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "particle":
        return ParticleNavigationFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "ukf":
        return UnscentedKalmanFilter(profile_obj, initial_position, initial_velocity)
    if filter_type == "ekf":
        return ExtendedKalmanFilter(profile_obj, initial_position, initial_velocity)
    raise ValueError(f"unknown filter type {filter_type!r}")


__all__ = [
    "AlphaBetaGammaFilter",
    "BaseKalmanFilter",
    "ExtendedKalmanFilter",
    "FilterProfile",
    "InformationKalmanFilter",
    "InnovationAdaptiveKalmanFilter",
    "LinearKalmanFilter",
    "NavFilter",
    "PROFILE_MAP",
    "PROFILES",
    "ParticleNavigationFilter",
    "Profile",
    "RobustHuberKalmanFilter",
    "SquareRootKalmanFilter",
    "UnscentedKalmanFilter",
    "create_filter",
    "get_profile",
]
