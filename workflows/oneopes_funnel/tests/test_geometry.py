"""Tests for oneopes_funnel.geometry (public BioSimSpace Funnel API only)."""

import math

import numpy as np
import pytest

from oneopes_funnel.geometry import orthonormal_basis

PROJECTIONS_NM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
P0 = np.array([1.0, 2.0, 3.0])
P1 = np.array([1.5, 0.8, 11.0])


# --------------------------------------------------------------------------
# Basis (no BioSimSpace)
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "axis",
    [[0.3, -1.2, 2.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -4.0], [1.0, 1.0, 0.0], [1e-3, 1.0, 1e-9]],
)
def test_orthonormal_basis_is_right_handed_and_orthonormal(axis):
    axis = np.asarray(axis, dtype=float)
    unit = axis / np.linalg.norm(axis)
    u, v = orthonormal_basis(axis)
    frame = np.array([u, v, unit])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(u, v), unit, atol=1e-12)


def test_orthonormal_basis_rejects_zero_and_non_finite_axis():
    with pytest.raises(ValueError, match="zero-length"):
        orthonormal_basis(np.zeros(3))
    with pytest.raises(ValueError, match="finite"):
        orthonormal_basis(np.array([1.0, np.nan, 0.0]))


# --------------------------------------------------------------------------
# Funnel CV parameters / radius (public Funnel API)
# --------------------------------------------------------------------------
@pytest.fixture
def default_cv():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    return Funnel([0, 1], [2, 3])


def test_parameters_equal_funnel_getters(default_cv):
    from oneopes_funnel.geometry import funnel_parameters

    cv = default_cv
    p = funnel_parameters(cv)
    assert p.width_nm == cv.getWidth().nanometers().value()
    assert p.buffer_nm == cv.getBuffer().nanometers().value()
    assert p.steepness_per_nm == cv.getSteepness()
    assert p.inflection_nm == cv.getInflection().nanometers().value()
    assert p.lower_bound_nm == cv.getLowerBound().getValue().nanometers().value()
    assert p.upper_bound_nm == cv.getUpperBound().getValue().nanometers().value()
    assert p.lower_bound_force_constant == float(cv.getLowerBound().getForceConstant())
    assert p.upper_bound_force_constant == float(cv.getUpperBound().getForceConstant())


def test_extent_equals_funnel_getextent(default_cv):
    from BioSimSpace.Types import Length

    from oneopes_funnel.geometry import funnel_extent

    expected = [default_cv.getExtent(Length(x, "nanometer")).nanometers().value() for x in PROJECTIONS_NM]
    np.testing.assert_array_equal(funnel_extent(default_cv, PROJECTIONS_NM), expected)


def test_extent_follows_the_documented_funnel_profile(default_cv):
    from oneopes_funnel.geometry import funnel_extent, funnel_parameters

    p = funnel_parameters(default_cv)
    profile = [p.width_nm / (1 + math.exp(p.steepness_per_nm * (s - p.inflection_nm))) + p.buffer_nm for s in PROJECTIONS_NM]
    np.testing.assert_allclose(funnel_extent(default_cv, PROJECTIONS_NM), profile, atol=1e-12)


def test_parameters_are_nm_whatever_the_input_unit():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics import Bound
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel
    from BioSimSpace.Types import Length

    from oneopes_funnel.geometry import funnel_parameters

    cv = Funnel(
        [0, 1], [2, 3],
        width=Length(6, "angstrom"), buffer=Length(1.5, "angstrom"), inflection=Length(20, "angstrom"),
        lower_bound=Bound(Length(5, "angstrom"), force_constant=2000),
        upper_bound=Bound(Length(40, "angstrom"), force_constant=2000),
    )
    p = funnel_parameters(cv)
    np.testing.assert_allclose(
        [p.width_nm, p.buffer_nm, p.inflection_nm, p.lower_bound_nm, p.upper_bound_nm], [0.6, 0.15, 2.0, 0.5, 4.0], atol=1e-12
    )


# --------------------------------------------------------------------------
# funnel_wall_points
# --------------------------------------------------------------------------
def _axial_radial(points, p0, p1):
    unit = (p1 - p0) / np.linalg.norm(p1 - p0)
    rel = points - p0
    proj = rel @ unit
    return proj, np.linalg.norm(rel - np.outer(proj, unit), axis=1)


@pytest.mark.parametrize("step_A, n_angles", [(2.0, 8), (2.5, 12), (7.0, 3)])
def test_wall_points_shape_and_finiteness(default_cv, step_A, n_angles):
    from oneopes_funnel.geometry import funnel_parameters, funnel_wall_points

    p = funnel_parameters(default_cv)
    points = funnel_wall_points(P0, P1, default_cv, step_A=step_A, n_angles=n_angles)
    n_rings = int(np.floor((10 * (p.upper_bound_nm - p.lower_bound_nm)) / step_A + 1e-9)) + 1
    assert isinstance(points, np.ndarray) and points.dtype == float
    assert points.shape == (n_rings * n_angles, 3)
    assert np.all(np.isfinite(points))


def test_wall_points_sample_the_axis_between_the_cv_bounds(default_cv):
    from oneopes_funnel.geometry import funnel_parameters, funnel_wall_points

    p = funnel_parameters(default_cv)
    points = funnel_wall_points(P0, P1, default_cv, step_A=2.0, n_angles=8)
    proj, _ = _axial_radial(points, P0, P1)
    rings = proj.reshape(-1, 8)
    np.testing.assert_allclose(rings, rings[:, :1] * np.ones((1, 8)), atol=1e-9)  # each ring at one axial position
    s = rings[:, 0]
    np.testing.assert_allclose(np.diff(s), 2.0, atol=1e-9)
    assert s[0] == pytest.approx(10 * p.lower_bound_nm, abs=1e-9)
    assert s[-1] <= 10 * p.upper_bound_nm + 1e-9


def test_wall_radii_equal_getextent_at_each_axial_position(default_cv):
    from BioSimSpace.Types import Length

    from oneopes_funnel.geometry import funnel_wall_points

    points = funnel_wall_points(P0, P1, default_cv)
    proj, radial = _axial_radial(points, P0, P1)
    expected = [default_cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj]
    np.testing.assert_allclose(radial, expected, atol=1e-9)


def test_wall_ring_points_are_evenly_spaced_in_angle(default_cv):
    from oneopes_funnel.geometry import funnel_wall_points

    n = 8
    points = funnel_wall_points(P0, P1, default_cv, n_angles=n)
    ring = points[:n]
    centre = ring.mean(axis=0)
    unit = (P1 - P0) / np.linalg.norm(P1 - P0)
    np.testing.assert_allclose(centre, P0 + ((centre - P0) @ unit) * unit, atol=1e-9)  # centred on the axis
    radius = np.linalg.norm(ring[0] - centre)
    chords = np.linalg.norm(ring - np.roll(ring, -1, axis=0), axis=1)
    np.testing.assert_allclose(chords, 2 * radius * np.sin(np.pi / n), atol=1e-9)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(p0=P0, p1=P0.copy()), "zero length"),
        (dict(p0=P0, p1=np.array([np.nan, 0.0, 0.0])), "finite"),
        (dict(p0=np.array([1.0, 2.0]), p1=P1), "3-vector"),
        (dict(step_A=0.0), "step_A"),
        (dict(step_A=-1.0), "step_A"),
        (dict(step_A=float("inf")), "step_A"),
        (dict(n_angles=2), "n_angles"),
        (dict(n_angles=4.5), "n_angles"),
        (dict(n_angles=True), "n_angles"),
    ],
)
def test_wall_points_reject_invalid_input(default_cv, kwargs, match):
    from oneopes_funnel.geometry import funnel_wall_points

    args = dict(p0=P0, p1=P1, cv=default_cv, step_A=2.0, n_angles=8) | kwargs
    with pytest.raises(ValueError, match=match):
        funnel_wall_points(**args)


def test_wall_points_need_cv_bounds():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    from oneopes_funnel.geometry import funnel_wall_points

    with pytest.raises(ValueError, match="bounds"):
        funnel_wall_points(P0, P1, Funnel([0], [1], lower_bound=None, upper_bound=None, grid=None))


def test_wall_points_on_validation_system_use_getextent(ref):
    from BioSimSpace.Types import Length

    from oneopes_funnel.make_funnel import make_funnel

    f = make_funnel(ref.system)
    points = f.wall_points()
    assert points.shape[1] == 3 and np.all(np.isfinite(points))
    proj, radial = _axial_radial(points, f.p0, f.p1)
    expected = [f.cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj]
    np.testing.assert_allclose(radial, expected, atol=1e-9)
