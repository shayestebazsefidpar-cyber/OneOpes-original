"""
Funnel geometry around a BioSimSpace ``Funnel`` collective variable.

All CV quantities come from the public ``BioSimSpace.Metadynamics.
CollectiveVariable.Funnel`` API:

* :func:`funnel_parameters` - the CV's width, buffer, steepness, inflection
  and bounds, read through the ``Funnel`` getters (nm).
* :func:`funnel_extent` - the funnel radius at given projections along the
  axis, from ``Funnel.getExtent()`` (nm).

:func:`funnel_wall_points` builds points on the funnel wall for inspection or
plotting: rings perpendicular to the P0 -> P1 axis, sampled every ``step_A``
between the CV's lower and upper bounds, with ring radius
``Funnel.getExtent()`` at that axial position. Coordinates are in Angstrom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .p0_p1 import require_biosimspace

MIN_AXIS_LENGTH_A = 1e-6


@dataclass(frozen=True)
class FunnelParameters:
    """Parameters of a BioSimSpace ``Funnel`` CV, in nm (BioSimSpace's unit for the CV)."""

    width_nm: float
    buffer_nm: float
    steepness_per_nm: float
    inflection_nm: float
    lower_bound_nm: float | None
    lower_bound_force_constant: float | None
    upper_bound_nm: float | None
    upper_bound_force_constant: float | None


def funnel_parameters(cv) -> FunnelParameters:
    """Read the parameters of a BioSimSpace ``Funnel`` collective variable."""

    def _bound(b):
        if b is None:
            return None, None
        return b.getValue().nanometers().value(), float(b.getForceConstant())

    lo, lo_k = _bound(cv.getLowerBound())
    hi, hi_k = _bound(cv.getUpperBound())
    return FunnelParameters(
        width_nm=cv.getWidth().nanometers().value(),
        buffer_nm=cv.getBuffer().nanometers().value(),
        steepness_per_nm=float(cv.getSteepness()),
        inflection_nm=cv.getInflection().nanometers().value(),
        lower_bound_nm=lo,
        lower_bound_force_constant=lo_k,
        upper_bound_nm=hi,
        upper_bound_force_constant=hi_k,
    )


def funnel_extent(cv, projection_nm) -> np.ndarray:
    """Funnel radius (nm) at projection(s) along the axis (nm), via ``cv.getExtent()``."""
    require_biosimspace()
    from BioSimSpace.Types import Length

    proj = np.atleast_1d(np.asarray(projection_nm, dtype=float))
    return np.array([cv.getExtent(Length(float(p), "nanometer")).nanometers().value() for p in proj])


def _as_point(value, name: str) -> np.ndarray:
    point = np.asarray(value, dtype=float)
    if point.shape != (3,):
        raise ValueError(f"{name} must be a 3-vector, got shape {point.shape}.")
    if not np.all(np.isfinite(point)):
        raise ValueError(f"{name} must contain only finite coordinates, got {point}.")
    return point


def orthonormal_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two unit vectors ``(u, v)`` that, with ``axis``, form a right-handed orthonormal frame.

    ``u`` is built from the Cartesian unit vector least aligned with ``axis``
    (so the projection never degenerates), then ``v = axis x u``.
    """
    axis = _as_point(axis, "axis")
    norm = np.linalg.norm(axis)
    if norm < MIN_AXIS_LENGTH_A:
        raise ValueError("Cannot build a basis around a zero-length axis.")
    axis = axis / norm
    reference = np.zeros(3)
    reference[np.argmin(np.abs(axis))] = 1.0
    u = reference - (reference @ axis) * axis
    u /= np.linalg.norm(u)
    v = np.cross(axis, u)
    return u, v


def funnel_wall_points(p0, p1, cv, step_A: float = 2.0, n_angles: int = 8) -> np.ndarray:
    """Points on the funnel wall (Angstrom).

    Parameters
    ----------
    p0, p1 : 3-vectors (A)
        Funnel origin (P0) and the point fixing the axis direction (P1).
    cv : BioSimSpace Funnel
        Source of the axial range (lower/upper bounds) and the radius
        (``getExtent()``).
    step_A : float
        Spacing (A) of the rings along the axis, from the lower bound up to
        (and including, when it falls on the grid) the upper bound.
    n_angles : int
        Points per ring (>= 3), evenly spaced in angle.

    Returns
    -------
    (n_rings * n_angles, 3) array, ring by ring along the axis. A point at
    axial position ``s`` (distance from P0 along P0 -> P1) and angle ``t`` is
    ``p0 + s * axis + r(s) * (cos(t) * u + sin(t) * v)`` with ``r(s)`` from
    ``cv.getExtent()`` and ``(u, v)`` from :func:`orthonormal_basis`.
    """
    p0 = _as_point(p0, "p0")
    p1 = _as_point(p1, "p1")
    if isinstance(step_A, bool) or not np.isfinite(step_A) or step_A <= 0:
        raise ValueError(f"step_A must be a finite number > 0, got {step_A!r}.")
    if isinstance(n_angles, bool) or not isinstance(n_angles, (int, np.integer)) or n_angles < 3:
        raise ValueError(f"n_angles must be an integer >= 3, got {n_angles!r}.")
    axis_vec = p1 - p0
    length = np.linalg.norm(axis_vec)
    if length < MIN_AXIS_LENGTH_A:
        raise ValueError("p0 and p1 coincide: the funnel axis has zero length.")
    axis = axis_vec / length
    u, v = orthonormal_basis(axis)

    params = funnel_parameters(cv)
    if params.lower_bound_nm is None or params.upper_bound_nm is None:
        raise ValueError("The Funnel CV needs lower and upper bounds to define the wall's axial range.")
    lower_A, upper_A = 10.0 * params.lower_bound_nm, 10.0 * params.upper_bound_nm
    n_rings = int(np.floor((upper_A - lower_A) / step_A + 1e-9)) + 1
    s = lower_A + step_A * np.arange(n_rings)
    radii = 10.0 * funnel_extent(cv, s / 10.0)

    theta = 2.0 * np.pi * np.arange(n_angles) / n_angles
    ring = np.cos(theta)[:, None] * u + np.sin(theta)[:, None] * v  # (n_angles, 3)
    points = p0 + s[:, None, None] * axis + radii[:, None, None] * ring[None, :, :]
    return points.reshape(-1, 3)
