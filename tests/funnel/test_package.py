"""B: package-level checks that need no BioSimSpace."""

import dataclasses
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[2] / "src" / "oneopes" / "funnel"
MODULES = ["oneopes.funnel", "oneopes.funnel.bss", "oneopes.funnel.p0_p1", "oneopes.funnel.geometry",
           "oneopes.funnel.make_funnel"]


@pytest.mark.parametrize("name", MODULES)
def test_b1_modules_import_without_biosimspace(name, monkeypatch):
    monkeypatch.setitem(sys.modules, "BioSimSpace", None)  # simulate BSS not installed
    importlib.import_module(name)


def test_b4_result_dataclasses_are_frozen():
    from oneopes.funnel.geometry import Bound, FunnelParameters
    from oneopes.funnel.make_funnel import FunnelResult
    from oneopes.funnel.p0_p1 import P0P1Result

    r = P0P1Result(atoms0=[0], atoms1=[1], p0=np.array([0.0, 0.0, 0.0]), p1=np.array([0.0, 0.0, 3.0]))
    lower = Bound(position_nm=0.5, force_constant=2000.0)
    params = FunnelParameters(width_nm=0.6, buffer_nm=0.15, steepness_per_nm=1.5, inflection_nm=2.0,
                              lower_bound=lower, upper_bound=Bound(position_nm=4.0, force_constant=2000.0))
    f = FunnelResult(p0p1=r, cv=None, parameters=params)
    for obj, field in [(r, "atoms0"), (lower, "position_nm"), (params, "width_nm"), (f, "cv")]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, None)
    assert (f.p0p1.atoms0, f.p0p1.atoms1, f.p0p1.axis_length) == ([0], [1], 3.0)


def test_b7_clear_error_without_biosimspace(monkeypatch):
    from oneopes.funnel.make_funnel import make_funnel
    from oneopes.funnel.p0_p1 import make_p0_p1

    monkeypatch.setitem(sys.modules, "BioSimSpace", None)
    for fn in (make_p0_p1, make_funnel):
        with pytest.raises(ImportError, match=r"BioSimSpace .*conda"):
            fn(system=None)


def test_b8_no_system_specific_values_in_src():
    forbidden = re.compile(r"HSP90|BRD4|lig\d|/home/|Escritorio|OneOpes-original", re.IGNORECASE)
    hits = [f"{path.name}:{number}: {line.strip()}" for path in SRC.glob("*.py")
            for number, line in enumerate(path.read_text().splitlines(), 1) if forbidden.search(line)]
    assert hits == []
