"""
Ligand, water oxygens and (when present) protein of a system, by the same rules everywhere.

* Protein: MDAnalysis ``protein`` selection (standard residue names); may be empty.
* Heavy atom: not element H (also mass-repartitioned hydrogens) and not a massless virtual
  site (e.g. a lone pair). Elements missing from the topology are guessed from atom names.
* Water: residues with a known water residue name (or an explicit selection); its oxygens are
  the water atoms of element O.
* Ligand: an explicit selection, or the single residue name, other than protein and water,
  of residues with more than one atom. Several such residue names is an error (no guessing).
"""

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from MDAnalysis import AtomGroup, Universe

from oneopes.errors import InputError

logger = logging.getLogger(__name__)

HEAVY_ATOMS = "not element H and prop mass > 0"
"""MDAnalysis selection of heavy atoms: not hydrogen, and not a massless virtual site."""

KNOWN_WATER_RESNAMES = (
    "WAT", "SOL", "HOH", "TIP3", "TIP3P", "TIP4", "TIP4P", "TIP5", "TIP5P", "T3P", "T4P", "SPC", "SPCE",
)


class SystemDefinitionError(InputError):
    """The input system cannot be split unambiguously into protein / ligand / water."""


@dataclass(frozen=True)
class SystemOptions:
    """How to find the ligand and the water."""
    ligand_selection: str | None = None
    """MDAnalysis selection of the ligand (default: auto-detect)."""
    water_selection: str | None = None
    """MDAnalysis selection of the water (default: known water residue names)."""


@dataclass(frozen=True)
class SystemSelection:
    """0-based atom indices of each part of the system, so they apply to any universe with the same topology."""
    protein_heavy: np.ndarray
    ligand_heavy: np.ndarray
    water_oxygens: np.ndarray
    ligand_selection: str
    water_selection: str

    def protein(self, u: Universe) -> AtomGroup:
        """The protein heavy atoms in ``u``."""
        return u.atoms[self.protein_heavy]

    def ligand(self, u: Universe) -> AtomGroup:
        """The ligand heavy atoms in ``u``."""
        return u.atoms[self.ligand_heavy]

    def water(self, u: Universe) -> AtomGroup:
        """The water oxygens in ``u``."""
        return u.atoms[self.water_oxygens]


def load_universe(topology: Path, *coordinates: Path) -> Universe:
    """``topology`` with ``coordinates`` (a structure or trajectories, if any); raises InputError when
    MDAnalysis cannot read them, e.g. when a trajectory has another number of atoms than the topology."""
    try:
        return Universe(str(topology), *map(str, coordinates))
    except Exception as err:  # MDAnalysis raises ValueError, OSError, TypeError, ... for files it cannot use
        files = ", ".join(Path(path).name for path in (topology, *coordinates))
        raise InputError(f"cannot read {files}: {err}") from err


def ensure_elements(u: Universe) -> None:
    """Guess the elements from the atom names (MDAnalysis guesser) when the topology has none."""
    if not hasattr(u.atoms, "elements"):
        logger.warning("The topology has no elements: guessing them from the atom names.")
        u.guess_TopologyAttrs(to_guess=["elements"])


def heavy_atoms(atoms: AtomGroup) -> AtomGroup:
    """The heavy atoms of ``atoms`` (see :data:`HEAVY_ATOMS`)."""
    ensure_elements(atoms.universe)
    return atoms.select_atoms(HEAVY_ATOMS)


def find_water(u: Universe, water_selection: str | None) -> tuple[AtomGroup, str]:
    """The water atoms (of ``water_selection``, or of the known water residue names) and the selection used."""
    if water_selection:
        water = u.select_atoms(water_selection)
    else:
        water = u.select_atoms("resname " + " ".join(KNOWN_WATER_RESNAMES))
        water_selection = "resname " + " ".join(sorted(set(water.resnames)))
    if len(water) == 0:
        raise SystemDefinitionError("No water found; pass an explicit water selection.")
    return water, water_selection


def find_ligand(u: Universe, ligand_selection: str | None, protein: AtomGroup,
                water: AtomGroup) -> tuple[AtomGroup, str]:
    """The ligand atoms (of ``ligand_selection``, or detected) and the selection used."""
    if not ligand_selection:
        others = u.atoms - protein - water
        names = sorted({residue.resname for residue in others.residues if residue.atoms.n_atoms > 1})
        if len(names) != 1:
            raise SystemDefinitionError(
                f"Cannot identify the ligand automatically (candidate residue names: {names}); "
                "pass an explicit ligand selection."
            )
        ligand_selection = f"resname {names[0]}"
    ligand = u.select_atoms(ligand_selection)
    if len(ligand) == 0:
        raise SystemDefinitionError(f"Ligand selection {ligand_selection!r} is empty.")
    if ligand & protein:
        raise SystemDefinitionError("Ligand selection overlaps the protein.")
    return ligand, ligand_selection


def select_system(u: Universe, options: SystemOptions = SystemOptions()) -> SystemSelection:
    """Identify the protein, ligand and water oxygens in ``u`` (guessing its elements if it has none)."""
    ensure_elements(u)
    protein = u.select_atoms("protein")
    water, water_selection = find_water(u, options.water_selection)
    ligand, ligand_selection = find_ligand(u, options.ligand_selection, protein, water)
    ligand_heavy = heavy_atoms(ligand)
    if len(ligand_heavy) == 0:
        raise SystemDefinitionError(f"Ligand selection {ligand_selection!r} has no heavy atoms.")
    water_oxygens = water.select_atoms("element O")
    if len(water_oxygens) == 0:
        raise SystemDefinitionError(f"Water selection {water_selection!r} contains no oxygen atoms.")

    return SystemSelection(
        protein_heavy=heavy_atoms(protein).indices,
        ligand_heavy=ligand_heavy.indices,
        water_oxygens=water_oxygens.indices,
        ligand_selection=ligand_selection,
        water_selection=water_selection,
    )
