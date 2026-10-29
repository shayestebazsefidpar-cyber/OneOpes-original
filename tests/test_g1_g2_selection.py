"""G1/G2 result types, their YAML file, and the `oneopes select` command on a synthetic system."""
import json

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml
from pydantic import ValidationError

from ligand_systems import ligand_in_water
from oneopes.cli import main
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import AtomRef, SelectionResult, select_g1_g2, write_g1_g2_yaml
from oneopes.ligand.select import ligand_atoms

RAW = {
    "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
    "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
}


def test_selection_result_dict_roundtrip():
    result = SelectionResult.model_validate(RAW)
    g1 = result.G1
    assert g1 == (AtomRef(name="X1", serial=3), AtomRef(name="X2", serial=7))
    assert result.model_dump(mode="json") == RAW


def test_selection_result_is_frozen():
    with pytest.raises(ValidationError):
        SelectionResult.model_validate(RAW).G1 = ()


@pytest.mark.parametrize("g1", [
    [{"name": "X1"}, {"name": "X2", "serial": 7}],                          # an atom without serial
    [{"name": "X1", "serial": 3}],                                          # one atom instead of a pair
    [{"name": "X1", "serial": "three"}, {"name": "X2", "serial": 7}],       # a serial that is not a number
])
def test_malformed_selection_is_rejected(g1):
    with pytest.raises(ValidationError):
        SelectionResult.model_validate({**RAW, "G1": g1})


def test_write_g1_g2_yaml_creates_parent_directory(tmp_path):
    out_path = tmp_path / "nested" / "g1_g2.yaml"
    write_g1_g2_yaml(SelectionResult.model_validate(RAW), out_path)
    assert yaml.safe_load(out_path.read_text()) == RAW


def chain(n_atoms: int) -> mda.AtomGroup:
    """Atoms C0-C1-...-C(n-1) bonded in a line, so atoms i and j are |i - j| bonds apart."""
    u = mda.Universe.empty(n_atoms)
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_atoms)])
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(n_atoms - 1)])
    return u.atoms


def test_rounded_fps_exactly_0_2_apart_are_different_fps():
    """In FP order C0, C1, C2, C5, C3, C4; C1 is bonded to C0, so G1's partner starts as C2.

    C5 (-0.1) is farther from C0 than C2 (-0.3), but 0.2 apart is not the same FP, so C2 stays.
    (The notebook's floating-point comparison counts -0.1 - -0.3 = 0.19999999999999998 as the same FP.)
    """
    atoms = chain(6)
    fp = np.array([-1.0, -0.5, -0.3, -0.04, 0.0, -0.1])
    selection = select_g1_g2(atoms, fp)
    assert [a.name for a in selection.G1] == ["C0", "C2"]
    assert [a.name for a in selection.G2] == ["C4", "C1"]


def test_farther_partner_with_the_same_fp_wins():
    """C2 and C5 share the rounded FP -0.3; C5 is farther from C0, so it becomes the partner."""
    fp = np.array([-1.0, -0.5, -0.31, -0.04, 0.0, -0.29])
    assert [a.name for a in select_g1_g2(chain(6), fp).G1] == ["C0", "C5"]


def test_atoms_not_bonded_into_one_molecule_are_rejected():
    u = mda.Universe.empty(3)
    u.add_TopologyAttr("names", ["C0", "C1", "C2"])
    u.add_TopologyAttr("bonds", [(0, 1)])
    with pytest.raises(InputError, match=r"not one bonded molecule but 2: C0, C1 \| C2"):
        select_g1_g2(u.atoms, np.array([-1.0, -0.5, 0.0]))


@pytest.mark.parametrize("names, indices", [(["C0", "C2"], [0, 1]), (["C0", "C1"], [0, 6])])
def test_ligand_atoms_must_be_the_measured_ones(names, indices):
    u = chain(6).universe
    hydration = xr.Dataset(coords={"atom": names, "atom_index": ("atom", indices)})
    assert list(ligand_atoms(u, hydration.isel(atom=[0])).indices) == [0]
    with pytest.raises(InputError, match="topology of the run"):
        ligand_atoms(u, hydration)


def test_select_command_writes_waterfp_and_g1_g2(tmp_path):
    top, traj = ligand_in_water(tmp_path)
    outdir = tmp_path / "selection"
    main(["select", "--tpr", top, "--xtc", traj, "--outdir", str(outdir)])

    fingerprints = pd.read_csv(outdir / "fingerprints.csv")
    assert list(fingerprints.columns) == ["atom", "fp"]
    assert list(pd.read_csv(outdir / "rdf.csv").columns) == ["atom", "r_nm", "n_r"]

    ligand = mda.Universe(top).select_atoms("resname MOL")
    expected = select_g1_g2(ligand, fingerprints["fp"].to_numpy())
    assert SelectionResult.model_validate(yaml.safe_load((outdir / "g1_g2.yaml").read_text())) == expected
    metadata = json.loads((outdir / "run_metadata.json").read_text())
    assert (metadata["frames"], metadata["ligand_selection"]) == ([0, 3], "resname MOL")
    assert set(metadata["inputs"]) == {"tpr", "xtc"}
