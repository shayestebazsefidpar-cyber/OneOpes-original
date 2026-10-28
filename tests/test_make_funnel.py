"""Tests for oneopes_funnel.make_funnel, against BioSimSpace makeFunnel() + Funnel."""

import numpy as np
import pytest
from conftest import bss_xyz

from oneopes_funnel.geometry import funnel_parameters


@pytest.fixture(scope="module")
def bss_reference(ref):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel, makeFunnel

    atoms0, atoms1 = makeFunnel(ref.system)
    return atoms0, atoms1, Funnel(atoms0, atoms1)


@pytest.fixture(scope="module")
def funnel(ref):
    from oneopes_funnel.make_funnel import make_funnel

    return make_funnel(ref.system)


def test_c7_cv_is_the_bss_funnel_built_from_makefunnel(funnel, bss_reference):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    atoms0, atoms1, _ = bss_reference
    assert isinstance(funnel.cv, Funnel)
    assert funnel.cv.getAtoms0() == list(atoms0)
    assert funnel.cv.getAtoms1() == list(atoms1)
    assert (funnel.atoms0, funnel.atoms1) == (list(atoms0), list(atoms1))


def test_c8_default_parameters_equal_bss_defaults(funnel, bss_reference):
    assert funnel.parameters == funnel_parameters(bss_reference[2])


def test_c9_funnel_kwargs_reach_bss_unchanged(ref):
    from BioSimSpace.Metadynamics import Bound
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel
    from BioSimSpace.Types import Length

    from oneopes_funnel.make_funnel import make_funnel

    kwargs = dict(
        width=Length(0.1, "nanometer"),
        buffer=Length(0.25, "nanometer"),
        steepness=0.5,
        inflection=Length(2.0, "nanometer"),
        lower_bound=Bound(Length(1.0, "nanometer"), force_constant=20000),
        upper_bound=Bound(Length(3.7, "nanometer"), force_constant=20000),
    )
    f = make_funnel(ref.system, **kwargs)
    reference = Funnel(f.atoms0, f.atoms1, **kwargs)
    assert f.parameters == funnel_parameters(reference)


def test_c10_extent_nm_equals_bss_getextent(funnel, bss_reference):
    from BioSimSpace.Types import Length

    proj = [0.0, 0.5, 1.0, 2.0, 3.0, 4.0]
    expected = [bss_reference[2].getExtent(Length(x, "nanometer")).nanometers().value() for x in proj]
    np.testing.assert_array_equal(funnel.extent_nm(proj), expected)


def test_c12_getcorrection_available_via_cv(funnel, bss_reference):
    from BioSimSpace.Types import Length

    assert funnel.cv.getCorrection().value() == bss_reference[2].getCorrection().value()
    kw = dict(proj_min=Length(0.5, "nanometer"), proj_max=Length(3.0, "nanometer"), delta=Length(0.01, "nanometer"))
    assert funnel.cv.getCorrection(**kw).value() == bss_reference[2].getCorrection(**kw).value()


def test_e3_p0_p1_in_angstrom_from_bss_coordinates(ref, funnel, bss_reference):
    atoms0, atoms1, _ = bss_reference
    np.testing.assert_allclose(funnel.p0, np.mean([bss_xyz(ref.system.getAtom(i)) for i in atoms0], axis=0), atol=1e-9)
    np.testing.assert_allclose(funnel.p1, np.mean([bss_xyz(ref.system.getAtom(i)) for i in atoms1], axis=0), atol=1e-9)
