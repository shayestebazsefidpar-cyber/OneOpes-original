"""
The candidate pool: the protein atoms the protein analysis follows.

Candidates are the protein heavy atoms whose minimum distance to any ligand heavy
atom in the complex structure is <= the cutoff (minimum image when the structure
has a box). The pool is computed once and never changes.
"""

import pandas as pd
import xarray as xr
from MDAnalysis.core.groups import Atom
from MDAnalysis.lib.distances import distance_array

from oneopes.protein.inputs import ProteinSystem
from oneopes.system import SystemDefinitionError

CANDIDATE_CUTOFF_A_DEFAULT = 7.0


def atom_label(atom: Atom) -> str:
    """The atom's name in the tables, e.g. ASP93_OD1_id1402 (residue, atom name, 1-based serial)."""
    return f"{atom.resname}{atom.resid}_{atom.name}_id{atom.index + 1}"


def select_candidates(system: ProteinSystem, cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT) -> pd.DataFrame:
    """Protein heavy atoms within ``cutoff_A`` of any ligand heavy atom in the complex structure.

    ``Atom_serial`` is the atom's 1-based number in the topology (``Atom_index`` + 1), as GROMACS
    counts atoms. ``Chain_ID`` comes only from the complex file's chain IDs (empty when
    unknown), never from the topology's chain field. Distances use the minimum image when the
    structure has a box.
    """
    u = system.complex
    protein, ligand = system.selection.protein(u), system.selection.ligand(u)
    min_distance = distance_array(protein.positions, ligand.positions, box=u.dimensions).min(axis=1)
    near = min_distance <= cutoff_A
    if not near.any():
        raise SystemDefinitionError(f"No protein heavy atom lies within {cutoff_A} A of the ligand.")
    atoms = protein[near]
    return pd.DataFrame({
        "Atom": [atom_label(atom) for atom in atoms],
        "Atom_index": atoms.indices,
        "Atom_serial": atoms.indices + 1,
        "Atom_name": atoms.names,
        "Residue_name": atoms.resnames,
        "Residue_number": atoms.resids,
        "Chain_ID": "" if system.chain_ids is None else system.chain_ids[atoms.indices],
        "Min_distance_to_ligand_A": min_distance[near],
    })


def candidate_details(candidates: pd.DataFrame) -> xr.Dataset:
    """The columns of :func:`select_candidates` along ``atom``, named by the ``Atom`` label, to attach to
    measurements as coordinates; ``Atom_index`` is left out (measurements carry it as ``atom_index``)."""
    return xr.Dataset.from_dataframe(candidates.drop(columns="Atom_index").set_index("Atom").rename_axis("atom"))
