"""
The candidate pool: the protein atoms the protein analysis follows.

Candidates are the protein heavy atoms whose minimum distance to any ligand heavy
atom in the complex structure is <= the cutoff (minimum image when the structure
has a box). The pool is computed once and never changes.
"""

import numpy as np
import xarray as xr
from MDAnalysis.core.groups import Atom
from MDAnalysis.lib.distances import distance_array

from oneopes.protein.inputs import ProteinSystem
from oneopes.system import SystemDefinitionError

CANDIDATE_CUTOFF_A_DEFAULT = 7.0


def atom_label(atom: Atom) -> str:
    """The atom's name in the tables, e.g. ASP93_OD1_id1402 (residue, atom name, 1-based serial)."""
    return f"{atom.resname}{atom.resid}_{atom.name}_id{atom.index + 1}"


def select_candidates(system: ProteinSystem, cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT) -> xr.Dataset:
    """Protein heavy atoms within ``cutoff_A`` of the ligand in the complex structure, as a Dataset of coordinates.

    ``Chain_ID`` comes only from the complex file: a GROMACS topology puts molecule-type names there.
    """
    u = system.complex
    protein, ligand = system.selection.protein(u), system.selection.ligand(u)
    min_distance = distance_array(protein.positions, ligand.positions, box=u.dimensions).min(axis=1)
    near = min_distance <= cutoff_A
    if not near.any():
        raise SystemDefinitionError(f"No protein heavy atom lies within {cutoff_A} A of the ligand.")
    atoms = protein[near]
    chain_ids = np.full(len(atoms), "", dtype=object) if system.chain_ids is None else system.chain_ids[atoms.indices]
    return xr.Dataset(coords={
        "atom": [atom_label(atom) for atom in atoms],
        "atom_index": ("atom", atoms.indices),
        "Atom_serial": ("atom", atoms.indices + 1),
        "Atom_name": ("atom", atoms.names),
        "Residue_name": ("atom", atoms.resnames),
        "Residue_number": ("atom", atoms.resids),
        "Chain_ID": ("atom", chain_ids),
        "Min_distance_to_ligand_A": ("atom", min_distance[near]),
    })
