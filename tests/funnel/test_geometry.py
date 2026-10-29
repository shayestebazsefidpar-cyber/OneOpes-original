"""Tests for oneopes.funnel.geometry (public BioSimSpace Funnel API only)."""

import math

import numpy as np
import pytest

from oneopes.funnel.p0_p1 import P0P1Result

PROJECTIONS_NM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
P0 = np.array([1.0, 2.0, 3.0])
P1 = np.array([1.5, 0.8, 11.0])
P0P1 = P0P1Result(atoms0=(0,), atoms1=(1,), p0=P0, p1=P1)


# --------------------------------------------------------------------------
# Funnel CV parameters / radius (public Funnel API)
# --------------------------------------------------------------------------
@pytest.fixture
def default_cv():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    return Funnel([0, 1], [2, 3])


def test_parameters_equal_funnel_getters(default_cv):
    from oneopes.funnel.geometry import funnel_parameters

    cv = default_cv
    p = funnel_parameters(cv)
    assert p.width_nm == cv.getWidth().nanometers().value()
    assert p.buffer_nm == cv.getBuffer().nanometers().value()
    assert p.steepness_per_nm == cv.getSteepness()
    assert p.inflection_nm == cv.getInflection().nanometers().value()
    assert p.lower_bound.position_nm == cv.getLowerBound().getValue().nanometers().value()
    assert p.upper_bound.position_nm == cv.getUpperBound().getValue().nanometers().value()
    assert p.lower_bound.force_constant == float(cv.getLowerBound().getForceConstant())
    assert p.upper_bound.force_constant == float(cv.getUpperBound().getForceConstant())


def test_extent_equals_funnel_getextent(default_cv):
    from BioSimSpace.Types import Length

    from oneopes.funnel.geometry import funnel_extent

    expected = [default_cv.getExtent(Length(x, "nanometer")).nanometers().value() for x in PROJECTIONS_NM]
    np.testing.assert_array_equal(funnel_extent(default_cv, PROJECTIONS_NM), expected)


def test_extent_follows_the_documented_funnel_profile(default_cv):
    from oneopes.funnel.geometry import funnel_extent, funnel_parameters

    p = funnel_parameters(default_cv)
    profile = [p.width_nm / (1 + math.exp(p.steepness_per_nm * (s - p.inflection_nm))) + p.buffer_nm
               for s in PROJECTIONS_NM]
    np.testing.assert_allclose(funnel_extent(default_cv, PROJECTIONS_NM), profile, atol=1e-12)


def test_parameters_are_nm_whatever_the_input_unit():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics import Bound
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel
    from BioSimSpace.Types import Length

    from oneopes.funnel.geometry import funnel_parameters

    cv = Funnel(
        [0, 1], [2, 3],
        width=Length(6, "angstrom"), buffer=Length(1.5, "angstrom"), inflection=Length(20, "angstrom"),
        lower_bound=Bound(Length(5, "angstrom"), force_constant=2000),
        upper_bound=Bound(Length(40, "angstrom"), force_constant=2000),
    )
    p = funnel_parameters(cv)
    np.testing.assert_allclose(
        [p.width_nm, p.buffer_nm, p.inflection_nm, p.lower_bound.position_nm, p.upper_bound.position_nm],
        [0.6, 0.15, 2.0, 0.5, 4.0], atol=1e-12
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
    from oneopes.funnel.geometry import funnel_parameters, funnel_wall_points

    p = funnel_parameters(default_cv)
    points = funnel_wall_points(P0P1, default_cv, step_A=step_A, n_angles=n_angles)
    length_A = 10 * (p.upper_bound.position_nm - p.lower_bound.position_nm)
    n_rings = int(np.floor(length_A / step_A + 1e-9)) + 1
    assert isinstance(points, np.ndarray) and points.dtype == float
    assert points.shape == (n_rings * n_angles, 3)
    assert np.all(np.isfinite(points))


def test_wall_points_sample_the_axis_between_the_cv_bounds(default_cv):
    from oneopes.funnel.geometry import funnel_parameters, funnel_wall_points

    p = funnel_parameters(default_cv)
    points = funnel_wall_points(P0P1, default_cv, step_A=2.0, n_angles=8)
    proj, _ = _axial_radial(points, P0, P1)
    rings = proj.reshape(-1, 8)
    np.testing.assert_allclose(rings, rings[:, :1] * np.ones((1, 8)), atol=1e-9)  # each ring at one axial position
    s = rings[:, 0]
    np.testing.assert_allclose(np.diff(s), 2.0, atol=1e-9)
    assert s[0] == pytest.approx(10 * p.lower_bound.position_nm, abs=1e-9)
    assert s[-1] <= 10 * p.upper_bound.position_nm + 1e-9


def test_wall_radii_equal_getextent_at_each_axial_position(default_cv):
    from BioSimSpace.Types import Length

    from oneopes.funnel.geometry import funnel_wall_points

    points = funnel_wall_points(P0P1, default_cv)
    proj, radial = _axial_radial(points, P0, P1)
    expected = [default_cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj]
    np.testing.assert_allclose(radial, expected, atol=1e-9)


@pytest.mark.parametrize("kwargs", [{"step_A": 0.0}, {"step_A": -1.0}, {"n_angles": 2}])
def test_wall_points_reject_invalid_input(default_cv, kwargs):
    from oneopes.funnel.geometry import funnel_wall_points

    with pytest.raises(ValueError, match="step_A > 0 and n_angles >= 3"):
        funnel_wall_points(P0P1, default_cv, **kwargs)


def test_wall_points_need_cv_bounds():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    from oneopes.funnel.geometry import funnel_wall_points

    with pytest.raises(ValueError, match="bounds"):
        funnel_wall_points(P0P1, Funnel([0], [1], lower_bound=None, upper_bound=None, grid=None))


def test_wall_points_on_validation_system_use_getextent(ref):
    from BioSimSpace.Types import Length

    from oneopes.funnel.geometry import funnel_wall_points
    from oneopes.funnel.make_funnel import make_funnel

    f = make_funnel(ref.system)
    points = funnel_wall_points(f.p0p1, f.cv)
    assert points.shape[1] == 3 and np.all(np.isfinite(points))
    proj, radial = _axial_radial(points, f.p0p1.p0, f.p0p1.p1)
    expected = [f.cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj]
    np.testing.assert_allclose(radial, expected, atol=1e-9)
