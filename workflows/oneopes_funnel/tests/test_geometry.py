"""Tests for oneopes_funnel.geometry, against the BioSimSpace Funnel CV and viewFunnel()."""

import inspect
import math

import numpy as np
import pytest
from conftest import bss_xyz, molecule_layout

PROJECTIONS_NM = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 4.5]
BSS_VIEWFUNNEL_LOOP_LINES = (1098, 1149)  # viewFunnel wall loop in BSS 2024.4.1 _funnel.py


# --------------------------------------------------------------------------
# B: wall-point basis (needs a BSS Funnel for the bounds/parameters)
# --------------------------------------------------------------------------
def test_b5_rings_are_perpendicular_circles_around_the_axis():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    from oneopes_funnel.geometry import funnel_wall_points

    p0, p1 = np.array([1.0, 2.0, 3.0]), np.array([1.5, 0.8, 11.0])
    unit = (p1 - p0) / np.linalg.norm(p1 - p0)
    points = funnel_wall_points(p0, p1, Funnel([0], [1]), basis_ints=(3, 7))
    for ring in points.reshape(-1, 8, 3):
        centre = ring.mean(axis=0)
        np.testing.assert_allclose((ring - centre) @ unit, 0.0, atol=1e-9)  # ring plane perpendicular to axis
        np.testing.assert_allclose(np.cross(centre - p0, unit), 0.0, atol=1e-9)  # ring centred on the axis


def test_b6_zero_axis_z_divides_by_zero_like_bss():
    # viewFunnel line 1127 divides by vec[2]; BSS then yields non-finite points
    # with a numpy RuntimeWarning (it does not raise) - reproduced verbatim.
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    from oneopes_funnel.geometry import funnel_wall_points

    with pytest.warns(RuntimeWarning):
        points = funnel_wall_points(np.zeros(3), np.array([1.0, 1.0, 0.0]), Funnel([0], [1]), basis_ints=(3, 7))
    assert not np.all(np.isfinite(points))


# --------------------------------------------------------------------------
# C / E: BSS Funnel CV (no system needed)
# --------------------------------------------------------------------------
@pytest.fixture
def default_cv():
    pytest.importorskip("BioSimSpace")
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    return Funnel([0, 1], [2, 3])


def test_c8_parameters_equal_bss_getters(default_cv):
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


def test_c10_extent_equals_bss_getextent(default_cv):
    from BioSimSpace.Types import Length

    from oneopes_funnel.geometry import funnel_extent

    expected = [default_cv.getExtent(Length(x, "nanometer")).nanometers().value() for x in PROJECTIONS_NM]
    np.testing.assert_array_equal(funnel_extent(default_cv, PROJECTIONS_NM), expected)


def test_c11_extent_matches_bss_formula(default_cv):
    from oneopes_funnel.geometry import funnel_extent, funnel_parameters

    p = funnel_parameters(default_cv)
    formula = [p.width_nm / (1 + math.exp(p.steepness_per_nm * (s - p.inflection_nm))) + p.buffer_nm for s in PROJECTIONS_NM]
    np.testing.assert_allclose(funnel_extent(default_cv, PROJECTIONS_NM), formula, atol=1e-12)
    np.testing.assert_allclose(funnel_extent(default_cv, p.inflection_nm), p.width_nm / 2 + p.buffer_nm, atol=1e-12)


def test_e1_parameters_are_nm_whatever_the_input_unit():
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


def test_e2_getextent_same_in_angstrom_and_nm(default_cv):
    from BioSimSpace.Types import Length

    for s_nm in PROJECTIONS_NM:
        in_nm = default_cv.getExtent(Length(s_nm, "nanometer")).nanometers().value()
        in_A = default_cv.getExtent(Length(10 * s_nm, "angstrom")).nanometers().value()
        assert in_A == pytest.approx(in_nm, abs=1e-12)


# --------------------------------------------------------------------------
# Wall points on the validation systems
# --------------------------------------------------------------------------
def _axial_radial(points, p0, p1):
    unit = (np.asarray(p1) - np.asarray(p0)) / np.linalg.norm(np.asarray(p1) - np.asarray(p0))
    rel = points - np.asarray(p0)
    proj = rel @ unit
    return proj, np.linalg.norm(rel - np.outer(proj, unit), axis=1)


def test_e4_default_wall_radius_is_getextent(ref):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel
    from BioSimSpace.Types import Length

    from oneopes_funnel.geometry import funnel_wall_points
    from oneopes_funnel.p0_p1 import make_p0_p1

    r = make_p0_p1(ref.system)
    cv = Funnel(r.atoms0, r.atoms1)
    proj, radial = _axial_radial(funnel_wall_points(r.p0, r.p1, cv, basis_ints=(3, 5)), r.p0, r.p1)
    expected = [cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj]
    np.testing.assert_allclose(radial, expected, atol=1e-9)


# --------------------------------------------------------------------------
# G: viewFunnel() characterization (real BSS function, run outside a notebook)
# --------------------------------------------------------------------------
def _run_viewfunnel(monkeypatch, system, cv, seed):
    """Call the real viewFunnel() and return its FUN pseudo-atom coordinates (A)."""
    import BioSimSpace.Metadynamics.CollectiveVariable._funnel as bss_funnel
    import BioSimSpace.Notebook as bss_notebook

    captured = {}
    monkeypatch.setattr(bss_funnel, "_is_notebook", True)
    monkeypatch.setattr(bss_notebook, "View", lambda s: captured.setdefault("system", s))
    np.random.seed(seed)
    bss_funnel.viewFunnel(system, cv)
    funnel_mol = captured["system"][-1]
    assert funnel_mol.getResidues()[0].name() == "FUN"
    return np.array([bss_xyz(a) for a in funnel_mol.getAtoms()])


def _viewfunnel_coms(system, atoms0, atoms1):
    """viewFunnel lines 1062-1082: COMs over the PROTEIN's own atom list."""
    protein, _, _, _ = molecule_layout(system)
    atoms = system[protein].getAtoms()
    return (np.mean([bss_xyz(atoms[i]) for i in atoms0], axis=0), np.mean([bss_xyz(atoms[i]) for i in atoms1], axis=0))


@pytest.fixture(scope="module")
def bss_cv(ref):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel, makeFunnel

    return Funnel(*makeFunnel(ref.system))


def test_g1_viewfunnel_uses_protein_local_indices(monkeypatch, ref, bss_cv):
    from oneopes_funnel.p0_p1 import make_p0_p1

    points = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=11)
    v0, v1 = _viewfunnel_coms(ref.system, bss_cv.getAtoms0(), bss_cv.getAtoms1())
    # viewFunnel's rings start at com0 + lower_bound * axis: recover its axis origin/direction
    lower_A = bss_cv.getLowerBound().getValue().angstroms().value()
    first_ring_centre = points[:8].mean(axis=0)
    unit = (v1 - v0) / np.linalg.norm(v1 - v0)
    np.testing.assert_allclose(first_ring_centre, v0 + lower_A * unit, atol=1e-6)

    ours = make_p0_p1(ref.system)
    protein, _, offsets, _ = molecule_layout(ref.system)
    if offsets[protein] == 0:
        np.testing.assert_allclose(v0, ours.p0, atol=1e-9)
        np.testing.assert_allclose(v1, ours.p1, atol=1e-9)
    else:  # protein is not the first molecule: viewFunnel reads shifted atoms
        assert np.linalg.norm(v0 - ours.p0) > 1e-3 or np.linalg.norm(v1 - ours.p1) > 1e-3


def test_g2_wall_loop_reproduces_real_viewfunnel(monkeypatch, ref, bss_cv):
    from oneopes_funnel.geometry import funnel_wall_points

    bss_points = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=1234)
    v0, v1 = _viewfunnel_coms(ref.system, bss_cv.getAtoms0(), bss_cv.getAtoms1())
    np.random.seed(1234)
    ours = funnel_wall_points(v0, v1, bss_cv, radius="viewfunnel")
    assert ours.shape == bss_points.shape
    np.testing.assert_allclose(ours, bss_points, atol=1e-6)


def test_g3_viewfunnel_radius_is_not_the_cv_radius(monkeypatch, ref, bss_cv):
    from BioSimSpace.Types import Length

    points = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=5)
    v0, v1 = _viewfunnel_coms(ref.system, bss_cv.getAtoms0(), bss_cv.getAtoms1())
    proj, radial = _axial_radial(points, v0, v1)
    width = bss_cv.getWidth().angstroms().value()
    buffer = bss_cv.getBuffer().angstroms().value()
    s_cent = bss_cv.getInflection().angstroms().value()
    beta = bss_cv.getSteepness()
    # viewFunnel line 1138: Angstrom distances, per-nm steepness
    np.testing.assert_allclose(radial, width / (1 + np.exp(beta * (proj - s_cent))) + buffer, atol=1e-6)
    cv_radius = np.array([bss_cv.getExtent(Length(float(s), "angstrom")).angstroms().value() for s in proj])
    at_inflection = np.isclose(proj, s_cent, atol=1e-6)
    np.testing.assert_allclose(radial[at_inflection], cv_radius[at_inflection], atol=1e-6)
    assert np.all(np.abs(radial[~at_inflection] - cv_radius[~at_inflection]) > 1e-3)


def test_g4_random_basis_only_rotates_rings(monkeypatch, ref, bss_cv):
    v0, v1 = _viewfunnel_coms(ref.system, bss_cv.getAtoms0(), bss_cv.getAtoms1())
    a = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=1)
    b = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=2)
    pa, ra = _axial_radial(a, v0, v1)
    pb, rb = _axial_radial(b, v0, v1)
    np.testing.assert_allclose(pa, pb, atol=1e-9)
    np.testing.assert_allclose(ra, rb, atol=1e-9)


def test_g5_ring_count_matches_viewfunnel(monkeypatch, ref, bss_cv):
    from oneopes_funnel.geometry import funnel_wall_points

    points = _run_viewfunnel(monkeypatch, ref.system, bss_cv, seed=3)
    lower = bss_cv.getLowerBound().getValue().angstroms().value()
    upper = bss_cv.getUpperBound().getValue().angstroms().value()
    assert len(points) == len(np.arange(lower, upper + 2, 2)) * 8
    assert len(funnel_wall_points(np.zeros(3), np.array([1.0, 1.0, 1.0]), bss_cv, basis_ints=(3, 5))) == len(points)


def test_g6_viewfunnel_returns_none_outside_notebook(ref, bss_cv):
    from BioSimSpace.Metadynamics.CollectiveVariable import viewFunnel

    assert viewFunnel(ref.system, bss_cv) is None


def test_g7_documented_divergences_from_viewfunnel_are_explicit_defaults():
    from oneopes_funnel.geometry import funnel_wall_points

    # radius: default is the CV/PLUMED radius (getExtent), viewFunnel's own is opt-in.
    assert inspect.signature(funnel_wall_points).parameters["radius"].default == "cv"
    with pytest.raises(ValueError):
        funnel_wall_points(np.zeros(3), np.ones(3), cv=None, radius="other")


def test_g8_wall_loop_is_verbatim_copy_of_installed_viewfunnel():
    """Only the documented ADAPTATION lines may differ from BSS viewFunnel lines 1098-1149."""
    bss = pytest.importorskip("BioSimSpace")
    from pathlib import Path

    import oneopes_funnel.geometry as geometry

    first, last = BSS_VIEWFUNNEL_LOOP_LINES
    bss_src = (Path(bss.__file__).parent / "Metadynamics/CollectiveVariable/_funnel.py").read_text().splitlines()
    bss_block = [line.strip() for line in bss_src[first - 1 : last] if line.strip()]
    ours = Path(geometry.__file__).read_text().splitlines()
    start = next(i for i, line in enumerate(ours) if "# ----- BEGIN verbatim" in line)
    end = next(i for i, line in enumerate(ours) if "# ----- END verbatim" in line)
    our_block = [line.strip() for line in ours[start + 1 : end] if line.strip() and "# ADAPTATION" not in line]

    # BSS lines replaced by the documented adaptations
    removed = {
        '# Get the element property from the map.',  # ADAPTATION 3 (pseudo-atoms)
        'element = property_map.get("element", "element")',  # ADAPTATION 3
        'funnel_coords.append(_SireVector(coord))',  # ADAPTATION 3 (numpy output)
    }
    assert [line for line in bss_block if line not in removed] == our_block
