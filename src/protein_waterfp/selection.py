"""
Protein / ligand / water identification and the candidate pool - all from
the current input system.

* Protein: MDAnalysis ``protein`` selection (standard residue names).
* Heavy atom: topology mass >= 1.5 and not element H (excludes hydrogens and
  massless virtual sites).
* Water: residues with a known water residue name (or an explicit
  selection); water oxygens are the oxygen atoms of those residues.
* Ligand: an explicit selection, or the single residue name that is not
  protein, water or a known ion and has more than one atom. Several such
  residue names is an error (no guessing).
* Candidates: protein heavy atoms whose minimum distance to any ligand heavy
  atom in the complex structure is <= the cutoff (minimum image when the
  structure has a box). The pool is computed once and never changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array

CANDIDATE_CUTOFF_A_DEFAULT = 7.0
HEAVY_ATOM_MIN_MASS = 1.5

KNOWN_WATER_RESNAMES = ("WAT", "SOL", "HOH", "TIP3", "TIP3P", "TIP4", "TIP4P", "TIP5", "TIP5P", "T3P", "T4P", "SPC", "SPCE")
KNOWN_ION_RESNAMES = ("NA", "CL", "NA+", "CL-", "SOD", "CLA", "K", "K+", "POT", "MG", "MG2+", "ZN", "ZN2+", "CA", "CA2+", "CAL")


class SystemDefinitionError(ValueError):
    """The input system cannot be split unambiguously into protein / ligand / water."""


@dataclass(frozen=True)
class SystemSelection:
    protein_heavy: np.ndarray
    """0-based indices of protein heavy atoms."""
    ligand_heavy: np.ndarray
    """0-based indices of ligand heavy atoms."""
    ligand_all: np.ndarray
    """0-based indices of all ligand atoms (for SASA)."""
    protein_all: np.ndarray
    """0-based indices of all protein atoms (for SASA)."""
    water_oxygens: np.ndarray
    ligand_description: str
    water_description: str


def heavy_mask(atoms) -> np.ndarray:
    """True for heavy atoms: mass >= 1.5 and not element H (when elements are known)."""
    mask = atoms.masses >= HEAVY_ATOM_MIN_MASS
    if hasattr(atoms, "elements"):
        mask &= np.array([str(e).strip().upper() != "H" for e in atoms.elements])
    return mask


def _water_atoms(u, water_selection):
    if water_selection:
        water = u.select_atoms(water_selection)
        description = water_selection
    else:
        names = [r for r in KNOWN_WATER_RESNAMES if len(u.select_atoms(f"resname {r}"))]
        water = u.select_atoms("resname " + " ".join(names)) if names else u.atoms[[]]
        description = "resname " + " ".join(names)
    if len(water) == 0:
        raise SystemDefinitionError("No water found; pass an explicit water selection.")
    return water, description


def select_system(u, ligand_selection: str | None = None, water_selection: str | None = None) -> SystemSelection:
    """Identify protein, ligand and water oxygens in ``u``."""
    protein = u.select_atoms("protein")
    if len(protein) == 0:
        raise SystemDefinitionError("The MDAnalysis 'protein' selection is empty.")
    water, water_desc = _water_atoms(u, water_selection)

    if ligand_selection:
        ligand = u.select_atoms(ligand_selection)
        ligand_desc = ligand_selection
    else:
        excluded = {r.upper() for r in KNOWN_WATER_RESNAMES + KNOWN_ION_RESNAMES}
        protein_res = set(protein.residues.ix)
        water_res = set(water.residues.ix)
        names = sorted({
            r.resname for r in u.residues
            if r.ix not in protein_res and r.ix not in water_res
            and r.resname.upper() not in excluded and len(r.atoms) > 1
        })
        if len(names) != 1:
            raise SystemDefinitionError(
                f"Cannot identify the ligand automatically (candidate residue names: {names}); "
                "pass an explicit ligand selection."
            )
        ligand = u.select_atoms(f"resname {names[0]}")
        ligand_desc = f"resname {names[0]}"
    if len(ligand) == 0:
        raise SystemDefinitionError(f"Ligand selection {ligand_desc!r} is empty.")
    if np.intersect1d(ligand.indices, protein.indices).size:
        raise SystemDefinitionError("Ligand selection overlaps the protein.")

    oxygen = np.array([str(n).upper().startswith("O") for n in water.names])
    water_o = water[oxygen]
    if len(water_o) == 0:
        raise SystemDefinitionError("Water selection contains no oxygen atoms (names starting with 'O').")

    return SystemSelection(
        protein_heavy=protein[heavy_mask(protein)].indices,
        ligand_heavy=ligand[heavy_mask(ligand)].indices,
        ligand_all=ligand.indices,
        protein_all=protein.indices,
        water_oxygens=water_o.indices,
        ligand_description=ligand_desc,
        water_description=water_desc,
    )


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
