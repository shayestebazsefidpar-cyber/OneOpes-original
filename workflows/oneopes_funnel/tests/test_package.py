"""B: package-level checks that need no BioSimSpace."""

import dataclasses
import importlib
import re
import sys
from pathlib import Path

import numpy as np
import pytest

SRC = Path(__file__).resolve().parents[1] / "src" / "oneopes_funnel"
MODULES = ["oneopes_funnel", "oneopes_funnel.p0_p1", "oneopes_funnel.geometry", "oneopes_funnel.make_funnel", "oneopes_funnel.plumed_io", "oneopes_funnel.system_io"]


@pytest.mark.parametrize("name", MODULES)
def test_b1_modules_import_without_biosimspace(name, monkeypatch):
    monkeypatch.setitem(sys.modules, "BioSimSpace", None)  # simulate BSS not installed
    importlib.import_module(name)


def test_b4_result_dataclasses_are_frozen():
    from oneopes_funnel.geometry import FunnelParameters
    from oneopes_funnel.make_funnel import FunnelResult
    from oneopes_funnel.p0_p1 import p0_p1_from_atoms

    r = p0_p1_from_atoms([0], [1], np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 3.0]]))
    params = FunnelParameters(0.6, 0.15, 1.5, 2.0, 0.5, 2000.0, 4.0, 2000.0)
    f = FunnelResult(p0p1=r, cv=None, parameters=params)
    for obj, field in [(r, "atoms0"), (params, "width_nm"), (f, "cv")]:
        with pytest.raises(dataclasses.FrozenInstanceError):
            setattr(obj, field, None)
    assert (f.atoms0, f.atoms1, f.axis_length) == ([0], [1], 3.0)


def test_b7_clear_error_without_biosimspace(monkeypatch):
    from oneopes_funnel.make_funnel import make_funnel
    from oneopes_funnel.p0_p1 import make_p0_p1

    monkeypatch.setitem(sys.modules, "BioSimSpace", None)
    for fn in (make_p0_p1, make_funnel):
        with pytest.raises(ImportError, match="BioSimSpace .*conda"):
            fn(system=None)


def test_b8_no_system_specific_values_in_src():
    forbidden = re.compile(r"HSP90|BRD4|lig\d|/home/|Escritorio|OneOpes-original", re.IGNORECASE)
    hits = [f"{p.name}:{n}: {line.strip()}" for p in SRC.glob("*.py") for n, line in enumerate(p.read_text().splitlines(), 1) if forbidden.search(line)]
    assert hits == []
