"""G1/G2 result types, their YAML file, and the ligand-waterfp-select command on a synthetic system."""
import dataclasses
import sys

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest
import yaml

from oneopes.ligand.g1_g2 import AtomRef, SelectionResult, main, select_g1_g2, write_g1_g2_yaml

RAW = {
    "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
    "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
}


def test_selection_result_dict_roundtrip():
    result = SelectionResult.from_dict(RAW)
    assert result.G1 == (AtomRef("X1", 3), AtomRef("X2", 7))
    assert result.to_yaml_dict() == RAW


def test_selection_result_is_frozen():
    with pytest.raises(dataclasses.FrozenInstanceError):
        SelectionResult.from_dict(RAW).G1 = ()


def test_atom_entry_missing_serial_is_rejected():
    with pytest.raises(KeyError):
        SelectionResult.from_dict({**RAW, "G1": [{"name": "X1"}, {"name": "X2", "serial": 7}]})


def test_write_g1_g2_yaml_creates_parent_directory(tmp_path):
    out_path = tmp_path / "nested" / "g1_g2.yaml"
    write_g1_g2_yaml(SelectionResult.from_dict(RAW), out_path)
    assert yaml.safe_load(out_path.read_text()) == RAW


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


def test_select_command_writes_waterfp_and_g1_g2(tmp_path, monkeypatch):
    top, traj = ligand_in_water(tmp_path)
    outdir = tmp_path / "selection"
    monkeypatch.setattr(sys, "argv", ["ligand-waterfp-select", "--tpr", top, "--xtc", traj, "--outdir", str(outdir)])
    main()

    fingerprints = pd.read_csv(outdir / "fingerprints.csv")
    assert list(fingerprints.columns) == ["atom", "fp"]
    assert list(pd.read_csv(outdir / "rdf.csv").columns) == ["atom", "r_nm", "n_r"]

    ligand = mda.Universe(top).select_atoms("resname MOL")
    expected = select_g1_g2(ligand, fingerprints["fp"].to_numpy())
    assert SelectionResult.from_dict(yaml.safe_load((outdir / "g1_g2.yaml").read_text())) == expected
