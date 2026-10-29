"""Fixtures for the protein-side tests."""

import pytest

from protein_systems import build_universe, simple_protein


@pytest.fixture
def small_system():
    """Protein chain along x, a 3-heavy-atom ligand near its middle, waters and one ion."""
    protein = simple_protein()
    ligand = [("C1", "C", (30.0, 16.0, 10.0)), ("N1", "N", (31.3, 16.0, 10.0)),
              ("O1", "O", (32.0, 17.1, 10.0)), ("H1", "H", (29.5, 16.8, 10.0))]
    ions = [("NA", [("NA", "Na", (50.0, 50.0, 50.0))])]
    return build_universe(protein, ligand, n_waters=150, extra_residues=ions)
