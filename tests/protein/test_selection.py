"""Candidate selection and the checks on the analysis inputs."""

import numpy as np
import pytest
from cyclopts import ValidationError

from oneopes.cli import app
from oneopes.protein import analysis
from oneopes.protein.candidates import CANDIDATE_CUTOFF_A_DEFAULT, select_candidates
from oneopes.protein.inputs import protein_system, validate_same_atoms
from oneopes.system import SystemDefinitionError, SystemOptions, select_system
from protein_systems import build_universe, protein_args, simple_protein


def _min_dist(u, sel, idx):
    lig = u.atoms[sel.ligand_heavy].positions
    return np.min(np.linalg.norm(lig - u.atoms[idx].position, axis=1))


def test_default_cutoff_is_exactly_7_A():
    assert CANDIDATE_CUTOFF_A_DEFAULT == 7.0
    assert analysis.AnalysisOptions().candidate_cutoff_A == 7.0


@pytest.mark.parametrize("value", [7.0, 10.0])
def test_explicit_cli_cutoff_overrides_default(tmp_path, value):
    _command, bound, _ = app.parse_args([*protein_args(tmp_path), "--candidate-cutoff-A", str(value)])
    assert bound.arguments["options"].candidate_cutoff_A == value


@pytest.mark.parametrize("value", ["0", "-3"])
def test_cli_rejects_a_cutoff_that_is_not_positive(tmp_path, value):
    with pytest.raises(ValidationError):
        app.parse_args([*protein_args(tmp_path), "--candidate-cutoff-A", value], exit_on_error=False,
                       print_error=False)


def test_a_system_without_protein_is_rejected():
    u = build_universe([], [("C1", "C", (12.0, 15.0, 10.0)), ("C2", "C", (13.0, 15.0, 10.0))], n_waters=5)
    with pytest.raises(SystemDefinitionError, match="no protein"):
        protein_system(u, u)


def test_atoms_within_cutoff_are_included_and_beyond_excluded(small_system):
    u = small_system
    system = protein_system(u, u)
    sel = system.selection
    cand = select_candidates(system)
    chosen = set(cand["Atom_index"])
    for i in sel.protein_heavy:
        d = _min_dist(u, sel, i)
        assert (i in chosen) == (d <= CANDIDATE_CUTOFF_A_DEFAULT), (i, d)
    assert 0 < len(chosen) < len(sel.protein_heavy)
    expected = [_min_dist(u, sel, i) for i in cand["Atom_index"]]
    np.testing.assert_allclose(cand["Min_distance_to_ligand_A"], expected, atol=1e-4)


def test_cutoff_is_a_parameter(small_system):
    system = protein_system(small_system, small_system)
    nearest = min(_min_dist(small_system, system.selection, i) for i in system.selection.protein_heavy)
    small = select_candidates(system, cutoff_A=nearest + 1.0)
    large = select_candidates(system, cutoff_A=nearest + 8.0)
    assert set(small["Atom_index"]) < set(large["Atom_index"])
    assert (small["Min_distance_to_ligand_A"] <= nearest + 1.0).all()


def test_pool_has_only_protein_heavy_atoms(small_system):
    u = small_system
    system = protein_system(u, u)
    cand = select_candidates(system, cutoff_A=50.0)
    atoms = u.atoms[cand["Atom_index"].to_numpy()]
    assert set(atoms.resnames) == {"ALA"}
    assert "H" not in set(atoms.elements) and (atoms.masses >= 1.5).all()
    assert not set(cand["Atom_index"]) & set(system.selection.ligand_all)


def test_seven_angstrom_only_as_the_configurable_default():
    """7 A appears only as CANDIDATE_CUTOFF_A_DEFAULT; nothing else hard-codes it."""
    import re
    from pathlib import Path

    src = Path(analysis.__file__).parent
    pattern = re.compile(r"(?<![\d.])7(\.0+)?\s*(A\b|Å|angstrom)|cutoff\w*\s*=\s*7(\.0+)?\b", re.IGNORECASE)
    hits = [f"{p.name}:{line.strip()}" for p in src.glob("*.py") for line in p.read_text().splitlines()
            if pattern.search(line)]
    assert hits == ["candidates.py:CANDIDATE_CUTOFF_A_DEFAULT = 7.0"]


def test_ambiguous_ligand_requires_explicit_selection():
    other = ("ZZY", [("X1", "C", (40.0, 40.0, 40.0)), ("X2", "C", (41.0, 40.0, 40.0))])
    u = build_universe(simple_protein(4), [("C1", "C", (12.0, 15.0, 10.0)), ("C2", "C", (13.0, 15.0, 10.0))],
                       n_waters=5, extra_residues=[other])
    with pytest.raises(SystemDefinitionError, match="ligand"):
        select_system(u)
    assert len(select_system(u, SystemOptions(ligand_selection="resname QZX")).ligand_heavy) == 2


def test_topology_atom_mismatch_fails(small_system):
    other = build_universe(simple_protein(3), [("C1", "C", (12.0, 15.0, 10.0))], n_waters=2)
    with pytest.raises(SystemDefinitionError, match="atoms"):
        validate_same_atoms(small_system, other, "complex")


def test_chain_id_never_taken_from_topology_molecule_type_names(small_system):
    """With a GROMACS .tpr, MDAnalysis puts molecule-type names (e.g. "Protein") in chainIDs."""
    small_system.atoms.chainIDs = ["Protein"] * len(small_system.atoms)
    cand = select_candidates(protein_system(small_system, small_system), 10.0)
    assert set(cand["Chain_ID"]) == {""}


def test_chain_id_taken_from_complex_structure_file(small_system, tmp_path):
    import MDAnalysis as mda

    small_system.atoms.chainIDs = ["K"] * len(small_system.atoms)
    small_system.atoms.write(tmp_path / "complex.pdb")
    small_system.atoms.chainIDs = ["Protein"] * len(small_system.atoms)
    complex_file = mda.Universe(str(tmp_path / "complex.pdb"))
    cand = select_candidates(protein_system(small_system, small_system, complex_file), 10.0)
    assert set(cand["Chain_ID"]) == {"K"}


def test_blank_chain_ids_in_structure_stay_empty(small_system):
    small_system.atoms.chainIDs = [" "] * len(small_system.atoms)
    cand = select_candidates(protein_system(small_system, small_system, small_system), 10.0)
    assert set(cand["Chain_ID"]) == {""}
