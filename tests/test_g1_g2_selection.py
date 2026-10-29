"""G1/G2 result types, their YAML file, and the `oneopes select` command on a synthetic system."""
import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
import yaml
from pydantic import ValidationError

from oneopes.cli import main
from oneopes.ligand.g1_g2 import AtomRef, SelectionResult, select_g1_g2, write_g1_g2_yaml

RAW = {
    "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
    "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
}


def test_selection_result_dict_roundtrip():
    result = SelectionResult.model_validate(RAW)
    assert result.G1 == (AtomRef(name="X1", serial=3), AtomRef(name="X2", serial=7))
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
    with pytest.raises(ValueError, match="not connected by bonds"):
        select_g1_g2(u.atoms, np.array([-1.0, -0.5, 0.0]))


def ligand_in_water(tmp_path, n_frames: int = 3) -> tuple[str, str]:
    """A 6-atom chain ligand in a 5 nm box of randomly placed water oxygens."""
    rng = np.random.default_rng(1)
    box, n_lig, n_water = 50.0, 6, 4000
    chain = np.array([[25.0 + 1.5 * i, 25.0, 25.0] for i in range(n_lig)])
    frames = [np.vstack([chain, rng.uniform(0.0, box, (n_water, 3))]) for _ in range(n_frames)]

    u = mda.Universe.empty(n_lig + n_water, n_residues=1 + n_water,
                           atom_resindex=[0] * n_lig + list(range(1, n_water + 1)), trajectory=True)
    u.add_TopologyAttr("names", [f"C{i}" for i in range(n_lig)] + ["O"] * n_water)
    u.add_TopologyAttr("resnames", ["MOL"] + ["SOL"] * n_water)
    u.add_TopologyAttr("bonds", [(i, i + 1) for i in range(n_lig - 1)])
    u.load_new(np.array(frames), order="fac", dimensions=[box, box, box, 90, 90, 90])

    top, traj = str(tmp_path / "top.pdb"), str(tmp_path / "traj.xtc")
    u.atoms.write(top)
    with mda.Writer(traj, u.atoms.n_atoms) as w:
        for _ in u.trajectory:
            w.write(u.atoms)
    return top, traj


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
