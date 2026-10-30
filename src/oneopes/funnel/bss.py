"""The BioSimSpace the funnel package uses, with its quantities as plain numbers (lengths in nm, positions in A)."""

import tempfile
from pathlib import Path
from types import ModuleType
from typing import Any

import MDAnalysis as mda
import numpy as np

from oneopes.errors import InputError

BssSystem = Any
"""A BioSimSpace ``System``; BioSimSpace has no type information, so these aliases only name the type."""
BssMolecule = Any
"""A BioSimSpace ``Molecule``."""
BssFunnel = Any
"""A BioSimSpace ``Funnel`` collective variable."""
BssBound = Any
"""A BioSimSpace ``Bound`` (a wall on a CV)."""
BssLength = Any
"""A BioSimSpace ``Length``."""
PropertyMap = dict[str, str]
"""BioSimSpace's mapping of property names, e.g. ``{"coordinates": "coordinates1"}``."""


def require_biosimspace() -> ModuleType:
    """Import BioSimSpace or fail with an actionable message."""
    try:
        import BioSimSpace  # pyright: ignore[reportMissingImports]
    except ImportError as err:
        raise ImportError(
            "oneopes.funnel needs BioSimSpace (conda-only, e.g. the 'biosimspace' "
            "conda env); it is not pip-installable."
        ) from err
    return BioSimSpace


def with_positions_of(coordinates: Path, structure: Path, out: Path) -> None:
    """Write ``coordinates`` to ``out`` with the atom positions of ``structure``; names, residues and box are kept."""
    u = mda.Universe(str(coordinates))
    positions = mda.Universe(str(structure)).atoms.positions
    if len(positions) != u.atoms.n_atoms:
        raise InputError(f"{structure} has {len(positions)} atoms, {coordinates} has {u.atoms.n_atoms}.")
    u.atoms.positions = positions
    u.atoms.write(str(out))


def read_system(topology: Path, coordinates: Path, structure: Path | None = None) -> BssSystem:
    """A BioSimSpace System from a GROMACS topology and its coordinates.

    With ``structure`` (e.g. a whole.pdb, which lacks the molecules), its positions replace the coordinates'.
    """
    bss = require_biosimspace()
    if structure is None:
        return bss.IO.readMolecules([str(topology), str(coordinates)])
    with tempfile.TemporaryDirectory() as tmp:
        positioned = Path(tmp) / Path(coordinates).name
        with_positions_of(Path(coordinates), Path(structure), positioned)
        return bss.IO.readMolecules([str(topology), str(positioned)])


def molecule_sizes(system: BssSystem) -> list[tuple[int, int]]:
    """(atoms, molecule index) of each molecule, largest first, as ``makeFunnel()`` picks protein and ligand."""
    return sorted(((molecule.nAtoms(), index) for index, molecule in enumerate(system.getMolecules())),
                  reverse=True)


def nanometers(length: BssLength) -> float:
    """A BioSimSpace ``Length`` in nm."""
    return length.nanometers().value()


def length_nm(value_nm: float) -> BssLength:
    """A BioSimSpace ``Length`` of ``value_nm`` nm."""
    require_biosimspace()
    from BioSimSpace.Types import Length  # pyright: ignore[reportMissingImports]

    return Length(float(value_nm), "nanometer")


def atom_position_A(system: BssSystem, index: int, property_map: PropertyMap | None = None) -> np.ndarray:
    """Position (A) of system atom ``index``, via the public ``Atom.coordinates()``."""
    position = system.getAtom(int(index)).coordinates(property_map=property_map or {})
    return np.array([position.x().angstroms().value(), position.y().angstroms().value(),
                     position.z().angstroms().value()])
