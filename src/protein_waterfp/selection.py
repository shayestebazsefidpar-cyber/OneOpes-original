"""
The candidate pool, and checks that two inputs describe the same atoms.

Candidates: protein heavy atoms whose minimum distance to any ligand heavy
atom in the complex structure is <= the cutoff (minimum image when the
structure has a box). The pool is computed once and never changes. The
protein, ligand and water themselves come from :mod:`system`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array

from system import SystemDefinitionError, SystemSelection

CANDIDATE_CUTOFF_A_DEFAULT = 7.0


def _box(u):
    dims = u.dimensions
    if dims is None or not np.all(np.isfinite(dims[:3])) or np.any(dims[:3] <= 0):
        return None
    return dims


def atom_label(atom) -> str:
    return f"{atom.resname}{atom.resid}_{atom.name}_id{atom.index + 1}"


def structure_chain_ids(u_structure) -> np.ndarray | None:
    """Chain IDs as written in a structure file (e.g. PDB column 22), one per atom.

    Pass a Universe built from the structure file alone: with a GROMACS .tpr
    topology MDAnalysis fills ``chainIDs`` with molecule-type names (e.g.
    "Protein"), which are not chain IDs. Returns None when the file carries no
    chain IDs; blank entries stay empty. Nothing is inferred.
    """
    if not hasattr(u_structure.atoms, "chainIDs"):
        return None
    return np.array([str(c or "").strip() for c in u_structure.atoms.chainIDs], dtype=object)


def select_candidates(u, selection: SystemSelection, cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT,
                      chain_ids=None) -> pd.DataFrame:
    """Protein heavy atoms within ``cutoff_A`` of any ligand heavy atom in ``u``'s current frame.

    ``Chain_ID`` comes only from ``chain_ids`` (see :func:`structure_chain_ids`),
    indexed by atom index; it is left empty when that is None. ``u``'s own
    topology chain field is never used.
    """
    if not np.isfinite(cutoff_A) or cutoff_A <= 0:
        raise ValueError(f"cutoff_A must be a finite number > 0, got {cutoff_A!r}.")
    if len(selection.protein_heavy) == 0:
        raise SystemDefinitionError("The MDAnalysis 'protein' selection is empty.")
    prot = u.atoms[selection.protein_heavy]
    lig = u.atoms[selection.ligand_heavy]
    min_dist = distance_array(prot.positions, lig.positions, box=_box(u)).min(axis=1)
    keep = min_dist <= cutoff_A
    rows = []
    for atom, dmin in zip(prot[keep], min_dist[keep]):
        chain = "" if chain_ids is None else chain_ids[atom.index]
        rows.append({
            "Atom": atom_label(atom),
            "Atom_index": int(atom.index),
            "Atom_serial_pdb": int(atom.index) + 1,
            "Atom_name": atom.name,
            "Residue_name": atom.resname,
            "Residue_number": int(atom.resid),
            "Chain_ID": chain,
            "Min_distance_to_ligand_A": float(dmin),
        })
    if not rows:
        raise SystemDefinitionError(f"No protein heavy atom lies within {cutoff_A} A of the ligand.")
    return pd.DataFrame(rows)


def validate_same_atoms(reference, other, what: str, indices=None) -> None:
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
