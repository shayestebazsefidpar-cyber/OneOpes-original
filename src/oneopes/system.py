"""
Ligand, water oxygens and (when present) protein of a system, by the same rules everywhere.

* Protein: MDAnalysis ``protein`` selection (standard residue names); may be empty.
* Heavy atom: topology mass >= 1.5 and not element H (excludes hydrogens and
  massless virtual sites).
* Water: residues with a known water residue name (or an explicit
  selection); water oxygens are the atoms of those residues whose name starts with O.
* Ligand: an explicit selection, or the single residue name that is not
  protein, water or a known ion and has more than one atom. Several such
  residue names is an error (no guessing).
"""

from dataclasses import dataclass

import numpy as np
from MDAnalysis import AtomGroup, Universe

HEAVY_ATOM_MIN_MASS = 1.5

KNOWN_WATER_RESNAMES = (
    "WAT", "SOL", "HOH", "TIP3", "TIP3P", "TIP4", "TIP4P", "TIP5", "TIP5P", "T3P", "T4P", "SPC", "SPCE",
)
KNOWN_ION_RESNAMES = (
    "NA", "CL", "NA+", "CL-", "SOD", "CLA", "K", "K+", "POT", "MG", "MG2+", "ZN", "ZN2+", "CA", "CA2+", "CAL",
)


class SystemDefinitionError(ValueError):
    """The input system cannot be split unambiguously into protein / ligand / water."""


@dataclass(frozen=True)
class SystemSelection:
    """0-based atom indices of each part of the system; the protein arrays are empty without a protein."""
    protein_heavy: np.ndarray
    ligand_heavy: np.ndarray
    ligand_all: np.ndarray
    protein_all: np.ndarray
    water_oxygens: np.ndarray
    ligand_description: str
    water_description: str


def heavy_mask(atoms: AtomGroup) -> np.ndarray:
    """True for heavy atoms: mass >= 1.5 and not element H (when elements are known)."""
    mask = atoms.masses >= HEAVY_ATOM_MIN_MASS
    if hasattr(atoms, "elements"):
        mask &= np.array([str(e).strip().upper() != "H" for e in atoms.elements], dtype=bool)
    return mask


def _water_atoms(u: Universe, water_selection: str | None) -> tuple[AtomGroup, str]:
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


def _ligand_atoms(u: Universe, ligand_selection: str | None, protein: AtomGroup,
                  water: AtomGroup) -> tuple[AtomGroup, str]:
    if not ligand_selection:
        excluded = {r.upper() for r in KNOWN_WATER_RESNAMES + KNOWN_ION_RESNAMES}
        other_res = set(protein.residues.ix) | set(water.residues.ix)
        names = sorted({r.resname for r in u.residues
                        if r.ix not in other_res and r.resname.upper() not in excluded and len(r.atoms) > 1})
        if len(names) != 1:
            raise SystemDefinitionError(
                f"Cannot identify the ligand automatically (candidate residue names: {names}); "
                "pass an explicit ligand selection."
            )
        ligand_selection = f"resname {names[0]}"
    ligand = u.select_atoms(ligand_selection)
    if len(ligand) == 0:
        raise SystemDefinitionError(f"Ligand selection {ligand_selection!r} is empty.")
    if np.intersect1d(ligand.indices, protein.indices).size:
        raise SystemDefinitionError("Ligand selection overlaps the protein.")
    return ligand, ligand_selection


def select_system(u: Universe, ligand_selection: str | None = None,
                  water_selection: str | None = None) -> SystemSelection:
    """Identify the protein, ligand and water oxygens in ``u``."""
    protein = u.select_atoms("protein")
    water, water_description = _water_atoms(u, water_selection)
    ligand, ligand_description = _ligand_atoms(u, ligand_selection, protein, water)
    ligand_heavy = ligand[heavy_mask(ligand)]
    if len(ligand_heavy) == 0:
        raise SystemDefinitionError(f"Ligand selection {ligand_description!r} has no heavy atoms.")

    water_o = water[np.array([str(n).upper().startswith("O") for n in water.names])]
    if len(water_o) == 0:
        raise SystemDefinitionError("Water selection contains no oxygen atoms (names starting with 'O').")

    return SystemSelection(
        protein_heavy=protein[heavy_mask(protein)].indices,
        ligand_heavy=ligand_heavy.indices,
        ligand_all=ligand.indices,
        protein_all=protein.indices,
        water_oxygens=water_o.indices,
        ligand_description=ligand_description,
        water_description=water_description,
    )
