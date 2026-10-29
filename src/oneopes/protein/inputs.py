"""The inputs of the protein analysis: one protein-ligand system, checked to be consistent."""

from dataclasses import dataclass

import numpy as np
from MDAnalysis import Universe

from oneopes.system import SystemDefinitionError, SystemOptions, SystemSelection, select_system


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
                   options: SystemOptions = SystemOptions()) -> ProteinSystem:
    """Check that the inputs describe the same atoms, and find the protein (required), ligand and water.

    ``complex_file``, the complex file read without the topology, gives the chain IDs; only its protein and
    ligand atom names are checked, since water and hydrogen naming differ between formats.
    """
    validate_same_atoms(trajectory, complex, "complex vs trajectory topology")
    selection = select_system(trajectory, options)
    if len(selection.protein_heavy) == 0:
        raise SystemDefinitionError("The system has no protein (the MDAnalysis 'protein' selection is empty).")
    chain_ids = None
    if complex_file is not None:
        validate_same_atoms(trajectory, complex_file, "complex structure",
                            indices=np.concatenate([selection.protein_heavy, selection.ligand_heavy]))
        chain_ids = structure_chain_ids(complex_file)
    return ProteinSystem(trajectory=trajectory, complex=complex, selection=selection, chain_ids=chain_ids)


def structure_chain_ids(u_structure: Universe) -> np.ndarray | None:
    """Chain IDs as written in a structure file, one per atom; None when it has none.

    Pass a Universe of the structure file alone: from a .tpr, MDAnalysis gives molecule-type names instead.
    """
    if not hasattr(u_structure.atoms, "chainIDs"):
        return None
    return np.array([str(c or "").strip() for c in u_structure.atoms.chainIDs], dtype=object)


def validate_same_atoms(reference: Universe, other: Universe, what: str, indices: np.ndarray | None = None) -> None:
    """Raise SystemDefinitionError unless ``other`` has the atom count of ``reference`` and its names at ``indices``."""
    if len(other.atoms) != len(reference.atoms):
        raise SystemDefinitionError(
            f"{what}: {len(other.atoms)} atoms, but the topology has {len(reference.atoms)}."
        )
    ours = reference.atoms if indices is None else reference.atoms[indices]
    theirs = other.atoms if indices is None else other.atoms[indices]
    differ = theirs.names != ours.names
    if differ.any():
        first = ours[differ][0]
        raise SystemDefinitionError(
            f"{what}: atom names differ from the topology ({differ.sum()} atoms; first at index {first.index}: "
            f"{theirs[differ][0].name!r} vs {first.name!r})."
        )
