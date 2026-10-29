"""The candidate pool: the protein atoms the protein analysis follows, chosen once on the complex structure."""

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

    ``chain_id`` comes only from the complex file: a GROMACS topology puts molecule-type names there.
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
        "atom_serial": ("atom", atoms.indices + 1),
        "atom_name": ("atom", atoms.names),
        "residue_name": ("atom", atoms.resnames),
        "residue_number": ("atom", atoms.resids),
        "chain_id": ("atom", chain_ids),
        "min_distance_to_ligand_A": ("atom", min_distance[near]),
    })
