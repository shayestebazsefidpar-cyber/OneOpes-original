"""Checks of `oneopes funnel` that need no BioSimSpace."""

import sys

import pytest

from oneopes.errors import InputError
from oneopes.funnel.command import funnel, funnel_summary


@pytest.fixture
def inputs(tmp_path):
    topology, coordinates = tmp_path / "complex.top", tmp_path / "complex.gro"
    topology.write_text("")
    coordinates.write_text("")
    return topology, coordinates


def test_existing_outdir_is_refused(inputs, tmp_path):
    with pytest.raises(InputError, match="exists"):
        funnel(topology=inputs[0], coordinates=inputs[1], outdir=tmp_path)


def test_missing_python_is_named(inputs, tmp_path):
    with pytest.raises(InputError, match="--python"):
        funnel(topology=inputs[0], coordinates=inputs[1], outdir=tmp_path / "out", python="no-such-python")


def test_python_without_biosimspace_reports_its_output(inputs, tmp_path):
    try:
        import BioSimSpace  # noqa: F401  # pyright: ignore[reportMissingImports]
        pytest.skip("this Python has BioSimSpace")
    except ImportError:
        pass
    with pytest.raises(InputError, match="BioSimSpace"):
        funnel(topology=inputs[0], coordinates=inputs[1], outdir=tmp_path / "out", python=sys.executable)


def test_summary_lists_atoms_axis_and_parameters():
    bound = {"position_nm": 0.5, "force_constant": 2000.0}
    text = funnel_summary({
        "protein": {"molecule": 0, "atoms": 100}, "ligand": {"molecule": 1, "atoms": 10, "residue": "MOL"},
        "atoms0_serial": [5, 6], "atoms1_serial": [7], "p0_A": [0.0, 0.0, 0.0], "p1_A": [0.0, 0.0, 10.0],
        "axis_length_A": 10.0, "radius_nm": {"0.0": 0.7},
        "parameters": {"width_nm": 0.6, "buffer_nm": 0.15, "steepness_per_nm": 1.5, "inflection_nm": 2.0,
                       "lower_bound": bound, "upper_bound": None},
    })
    assert "ligand: molecule 1 (MOL), 10 atoms" in text
    assert "atoms0 (P0, origin) serials: [5, 6]" in text
    assert "axis length (A): 10.00" in text
    assert "upper bound: None" in text
