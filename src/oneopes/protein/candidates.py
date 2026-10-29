"""
The candidate pool: the protein atoms the protein analysis follows.

Candidates are the protein heavy atoms whose minimum distance to any ligand heavy
atom in the complex structure is <= the cutoff (minimum image when the structure
has a box). The pool is computed once and never changes.
"""

import numpy as np
import pandas as pd
from MDAnalysis import Universe
from MDAnalysis.lib.distances import distance_array

from oneopes.protein.inputs import ProteinSystem
from oneopes.system import SystemDefinitionError

CANDIDATE_CUTOFF_A_DEFAULT = 7.0


def _box(u: Universe) -> np.ndarray | None:
    dims = u.dimensions
    if dims is None or not np.all(np.isfinite(dims[:3])) or np.any(dims[:3] <= 0):
        return None
    return dims


def atom_label(atom) -> str:
    return f"{atom.resname}{atom.resid}_{atom.name}_id{atom.index + 1}"


def select_candidates(system: ProteinSystem, cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT) -> pd.DataFrame:
    """Protein heavy atoms within ``cutoff_A`` of any ligand heavy atom in the complex structure.

    ``Chain_ID`` comes only from the complex file's chain IDs (empty when unknown), never from
    the topology's chain field.
    """
    if not np.isfinite(cutoff_A) or cutoff_A <= 0:
        raise ValueError(f"cutoff_A must be a finite number > 0, got {cutoff_A!r}.")
    if len(system.selection.protein_heavy) == 0:
        raise SystemDefinitionError("The MDAnalysis 'protein' selection is empty.")
    u = system.complex
    protein = u.atoms[system.selection.protein_heavy]
    ligand = u.atoms[system.selection.ligand_heavy]
    min_distance = distance_array(protein.positions, ligand.positions, box=_box(u)).min(axis=1)
    near = min_distance <= cutoff_A
    if not near.any():
        raise SystemDefinitionError(f"No protein heavy atom lies within {cutoff_A} A of the ligand.")
    atoms = protein[near]
    return pd.DataFrame({
        "Atom": [atom_label(atom) for atom in atoms],
        "Atom_index": atoms.indices,
        "Atom_serial_pdb": atoms.indices + 1,
        "Atom_name": atoms.names,
        "Residue_name": atoms.resnames,
        "Residue_number": atoms.resids,
        "Chain_ID": "" if system.chain_ids is None else system.chain_ids[atoms.indices],
        "Min_distance_to_ligand_A": min_distance[near],
    })
