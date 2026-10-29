"""Tests for the shared `system` module on a protein-free ligand-in-water box."""
import MDAnalysis as mda
import numpy as np
import pytest

from oneopes.system import SystemDefinitionError, select_system

MASSES = {"C": 12.011, "O": 15.999, "H": 1.008, "Na": 22.99, "X": 0.0}


def ligand_in_water(ligand: list[tuple[str, str]], n_water: int = 3, ligand_resname: str = "MOL",
                    water_resname: str = "SOL", water_names: tuple[str, ...] = ("OW", "HW1", "HW2")) -> mda.Universe:
    """One ligand residue of (name, element) atoms, then ``n_water`` waters, plus a sodium ion."""
    water = [(name, name[0] if name[0] in "OH" else "X") for name in water_names]
    atoms = ligand + water * n_water + [("NA", "Na")]
    residues = [0] * len(ligand) + [1 + i // len(water) for i in range(n_water * len(water))] + [1 + n_water]
    u = mda.Universe.empty(len(atoms), n_residues=n_water + 2, atom_resindex=residues, trajectory=True)
    u.add_TopologyAttr("names", [name for name, _ in atoms])
    u.add_TopologyAttr("elements", [element for _, element in atoms])
    u.add_TopologyAttr("masses", [MASSES[element] for _, element in atoms])
    u.add_TopologyAttr("resnames", [ligand_resname] + [water_resname] * n_water + ["NA"])
    u.add_TopologyAttr("resids", list(range(1, n_water + 3)))
    return u


LIGAND = [("C1", "C"), ("H1", "H"), ("O1", "O"), ("H2", "H"), ("C2", "C")]


def test_ligand_in_water_without_protein():
    u = ligand_in_water(LIGAND)
    sel = select_system(u)
    assert list(u.atoms[sel.ligand_heavy].names) == ["C1", "O1", "C2"]
    assert sel.protein_heavy.size == 0 and sel.ligand_description == "resname MOL"
    assert list(u.atoms[sel.water_oxygens].names) == ["OW"] * 3


def test_heavy_atoms_are_the_ones_a_name_rule_would_pick_for_an_ordinary_ligand():
    u = ligand_in_water(LIGAND)
    assert np.array_equal(select_system(u).ligand_heavy, u.select_atoms("resname MOL and not name H*").indices)


def test_virtual_sites_are_not_heavy_atoms():
    u = ligand_in_water(LIGAND + [("LP1", "X")], water_names=("OW", "HW1", "HW2", "MW"))
    sel = select_system(u)
    assert list(u.atoms[sel.ligand_heavy].names) == ["C1", "O1", "C2"]
    assert list(u.atoms[sel.water_oxygens].names) == ["OW"] * 3


def test_explicit_selections_override_detection():
    u = ligand_in_water(LIGAND, water_resname="XWT", water_names=("O", "H1", "H2"))
    with pytest.raises(SystemDefinitionError, match="water"):
        select_system(u)
    sel = select_system(u, ligand_selection="resname MOL and name C*", water_selection="resname XWT")
    assert list(u.atoms[sel.ligand_heavy].names) == ["C1", "C2"]
    assert len(sel.water_oxygens) == 3


def test_ligand_without_heavy_atoms_is_an_error():
    with pytest.raises(SystemDefinitionError, match="no heavy atoms"):
        select_system(ligand_in_water(LIGAND), ligand_selection="resname MOL and name H*")
