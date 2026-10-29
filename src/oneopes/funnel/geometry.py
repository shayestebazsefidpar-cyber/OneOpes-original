"""The funnel of a BioSimSpace ``Funnel`` CV as plain numbers: its parameters, its radius and points on its wall."""

from dataclasses import dataclass

import numpy as np
from MDAnalysis.units import convert
from numpy.typing import ArrayLike

from oneopes.errors import InputError
from oneopes.funnel.bss import BssBound, BssFunnel, length_nm, nanometers
from oneopes.funnel.p0_p1 import FunnelAxis
from oneopes.vectors import circles_around_axis


@dataclass(frozen=True)
class Bound:
    """A harmonic wall on the funnel's axial coordinate."""

    position_nm: float
    force_constant: float


@dataclass(frozen=True)
class FunnelParameters:
    """Parameters of a BioSimSpace ``Funnel`` CV."""

    width_nm: float
    buffer_nm: float
    steepness_per_nm: float
    inflection_nm: float
    lower_bound: Bound | None
    upper_bound: Bound | None


def bound(cv_bound: BssBound | None) -> Bound | None:
    """A BioSimSpace ``Bound`` as plain numbers; None for no bound."""
    if cv_bound is None:
        return None
    return Bound(position_nm=nanometers(cv_bound.getValue()), force_constant=float(cv_bound.getForceConstant()))


def funnel_parameters(cv: BssFunnel) -> FunnelParameters:
    """The parameters of a BioSimSpace ``Funnel`` CV, read through its getters."""
    return FunnelParameters(
        width_nm=nanometers(cv.getWidth()),
        buffer_nm=nanometers(cv.getBuffer()),
        steepness_per_nm=float(cv.getSteepness()),
        inflection_nm=nanometers(cv.getInflection()),
        lower_bound=bound(cv.getLowerBound()),
        upper_bound=bound(cv.getUpperBound()),
    )


def funnel_extent(cv: BssFunnel, projection_nm: ArrayLike) -> np.ndarray:
    """Funnel radius (nm) at each projection (nm) along the axis, from ``cv.getExtent()``."""
    return np.array([nanometers(cv.getExtent(length_nm(p))) for p in np.atleast_1d(projection_nm)])


def funnel_wall_points(axis: FunnelAxis, cv: BssFunnel, step_A: float = 2.0, n_angles: int = 8) -> np.ndarray:
    """Points on the funnel wall (A): a ring every ``step_A`` along the axis between the CV's bounds, shape (n, 3)."""
    if step_A <= 0 or n_angles < 3:
        raise InputError(f"The wall needs step_A > 0 and n_angles >= 3, got {step_A} and {n_angles}.")
    parameters = funnel_parameters(cv)
    if parameters.lower_bound is None or parameters.upper_bound is None:
        raise InputError("The Funnel CV needs lower and upper bounds to define the wall's axial range.")
    lower_A = convert(parameters.lower_bound.position_nm, "nm", "A")
    upper_A = convert(parameters.upper_bound.position_nm, "nm", "A")
    positions_A = np.arange(lower_A, upper_A + 1e-9, step_A)
    radii_A = convert(funnel_extent(cv, convert(positions_A, "A", "nm")), "nm", "A")
    rings = circles_around_axis(axis.p0, axis.direction, positions_A, radii_A, n_angles)
    return rings.reshape(-1, 3)
