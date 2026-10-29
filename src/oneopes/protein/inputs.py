"""The inputs of the protein analysis: one protein-ligand system, checked to be consistent."""

from dataclasses import dataclass

import numpy as np
from MDAnalysis import Universe

from oneopes.system import SystemDefinitionError, SystemSelection, select_system


@dataclass(frozen=True)
class ProteinSystem:
    """A protein-ligand system: its trajectory, its complex structure, and which atoms are which."""
    trajectory: Universe
    """Topology and production trajectory."""
    complex: Universe
    """Topology and complex coordinates, for the candidate pool and SASA."""
    selection: SystemSelection
    chain_ids: np.ndarray | None
    """Chain IDs as written in the complex file, one per atom; None when unknown."""


def protein_system(trajectory: Universe, complex: Universe, complex_file: Universe | None = None,
                   ligand_selection: str | None = None, water_selection: str | None = None) -> ProteinSystem:
    """Check that the inputs describe the same atoms, and find the protein (required), ligand and water.

    ``complex_file`` is the complex file read on its own: its protein and ligand heavy-atom names must
    match the topology (water and hydrogen naming may differ between formats), and its chain IDs fill
    ``Chain_ID``.
    """
    validate_same_atoms(trajectory, complex, "complex vs trajectory topology")
    selection = select_system(trajectory, ligand_selection, water_selection)
    if len(selection.protein_heavy) == 0:
        raise SystemDefinitionError("The system has no protein (the MDAnalysis 'protein' selection is empty).")
    chain_ids = None
    if complex_file is not None:
        validate_same_atoms(trajectory, complex_file, "complex structure",
                            indices=np.concatenate([selection.protein_heavy, selection.ligand_heavy]))
        chain_ids = structure_chain_ids(complex_file)
    return ProteinSystem(trajectory=trajectory, complex=complex, selection=selection, chain_ids=chain_ids)


def structure_chain_ids(u_structure: Universe) -> np.ndarray | None:
    """Chain IDs as written in a structure file (e.g. PDB column 22), one per atom.

    Pass a Universe built from the structure file alone: with a GROMACS .tpr
    topology MDAnalysis fills ``chainIDs`` with molecule-type names (e.g.
    "Protein"), which are not chain IDs. Returns None when the file carries no
    chain IDs; blank entries stay empty. Nothing is inferred.
    """
    if not hasattr(u_structure.atoms, "chainIDs"):
        return None
    return np.array([str(c or "").strip() for c in u_structure.atoms.chainIDs], dtype=object)


def validate_same_atoms(reference: Universe, other: Universe, what: str, indices=None) -> None:
    """Fail unless ``other`` has the same atom count as ``reference`` and the same
    atom names at ``indices`` (all atoms if None)."""
    if len(other.atoms) != len(reference.atoms):
        raise SystemDefinitionError(
            f"{what}: {len(other.atoms)} atoms, but the topology has {len(reference.atoms)}."
        )
    idx = np.arange(len(reference.atoms)) if indices is None else np.asarray(indices)
    names_other = np.asarray(other.atoms.names)[idx]
    names_ref = np.asarray(reference.atoms.names)[idx]
    mismatch = idx[names_other != names_ref]
    if mismatch.size:
        i = int(mismatch[0])
        raise SystemDefinitionError(
            f"{what}: atom names differ from the topology ({mismatch.size} atoms; first at index {i}: "
            f"{other.atoms[i].name!r} vs {reference.atoms[i].name!r})."
        )
