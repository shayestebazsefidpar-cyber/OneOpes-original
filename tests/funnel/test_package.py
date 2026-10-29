"""Package-level checks of oneopes.funnel that need no BioSimSpace."""

import dataclasses
import importlib
import sys

import numpy as np
import pytest

MODULES = ["oneopes.funnel", "oneopes.funnel.bss", "oneopes.funnel.p0_p1", "oneopes.funnel.geometry",
           "oneopes.funnel.make_funnel"]


@pytest.mark.parametrize("name", MODULES)
def test_modules_import_without_biosimspace(name, monkeypatch):
    monkeypatch.setitem(sys.modules, "BioSimSpace", None)  # simulate BSS not installed
    importlib.import_module(name)


def test_result_dataclasses_are_frozen():
    from oneopes.funnel.geometry import Bound, FunnelParameters
    from oneopes.funnel.make_funnel import FunnelResult
    from oneopes.funnel.p0_p1 import FunnelAxis

    r = FunnelAxis(atoms0=(0,), atoms1=(1,), p0=np.array([0.0, 0.0, 0.0]), p1=np.array([0.0, 0.0, 3.0]))
    lower = Bound(position_nm=0.5, force_constant=2000.0)
    params = FunnelParameters(width_nm=0.6, buffer_nm=0.15, steepness_per_nm=1.5, inflection_nm=2.0,
                              lower_bound=lower, upper_bound=Bound(position_nm=4.0, force_constant=2000.0))
    f = FunnelResult(axis=r, cv=None)
    for obj, field in [(r, "atoms0"), (lower, "position_nm"), (params, "width_nm"), (f, "cv")]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, None)
    assert (f.axis.atoms0, f.axis.atoms1, f.axis.length) == ((0,), (1,), 3.0)


def test_clear_error_without_biosimspace(monkeypatch):
    from oneopes.funnel.make_funnel import make_funnel
    from oneopes.funnel.p0_p1 import funnel_axis

    monkeypatch.setitem(sys.modules, "BioSimSpace", None)
    for fn in (funnel_axis, make_funnel):
        with pytest.raises(ImportError, match=r"BioSimSpace .*conda"):
            fn(system=None)
