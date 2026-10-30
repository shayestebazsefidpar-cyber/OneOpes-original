"""Tests for oneopes.funnel.bss that need no BioSimSpace: taking positions from another structure."""

import MDAnalysis as mda
import numpy as np
import pytest

from oneopes.errors import InputError
from oneopes.funnel.bss import with_positions_of


def _structure(path, names, positions):
    """A one-residue structure file with the given atom names and positions (A)."""
    u = mda.Universe.empty(len(names), trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("resids", [1])
    u.atoms.positions = positions
    u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
    u.atoms.write(str(path))


def test_positions_come_from_the_structure_and_names_from_the_coordinates(tmp_path):
    _structure(tmp_path / "npt.gro", ["OW", "HW1", "HW2"], np.zeros((3, 3)))
    moved = np.array([[1.0, 2.0, 3.0], [4.5, 5.5, 6.5], [7.25, 8.25, 9.25]])
    _structure(tmp_path / "whole.pdb", ["O", "H1", "H2"], moved)

    with_positions_of(tmp_path / "npt.gro", tmp_path / "whole.pdb", tmp_path / "out.gro")

    out = mda.Universe(str(tmp_path / "out.gro"))
    assert list(out.atoms.names) == ["OW", "HW1", "HW2"]
    np.testing.assert_allclose(out.atoms.positions, moved, atol=0.01)  # .gro keeps 0.01 A


def test_the_atom_counts_must_match(tmp_path):
    _structure(tmp_path / "npt.gro", ["OW", "HW1", "HW2"], np.zeros((3, 3)))
    _structure(tmp_path / "whole.pdb", ["O", "H1"], np.zeros((2, 3)))
    with pytest.raises(InputError, match="2 atoms"):
        with_positions_of(tmp_path / "npt.gro", tmp_path / "whole.pdb", tmp_path / "out.gro")
