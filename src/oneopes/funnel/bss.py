"""The BioSimSpace the funnel package uses, with its quantities as plain numbers (lengths in nm, positions in A)."""

import tempfile
from pathlib import Path

import MDAnalysis as mda
import numpy as np

from oneopes.errors import InputError


def require_biosimspace():
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
    """Write ``coordinates`` (e.g. npt.gro) to ``out`` with the atom positions of ``structure``, matched by index.

    Names, residues and box stay those of ``coordinates``. Raises InputError when the atom counts differ.
    """
    u = mda.Universe(str(coordinates))
    positions = mda.Universe(str(structure)).atoms.positions
    if len(positions) != u.atoms.n_atoms:
        raise InputError(f"{structure} has {len(positions)} atoms, {coordinates} has {u.atoms.n_atoms}.")
    u.atoms.positions = positions
    u.atoms.write(str(out))


def read_system(topology: Path, coordinates: Path, structure: Path | None = None):
    """A BioSimSpace System from a GROMACS topology and its coordinates (e.g. top.top and npt.gro).

    With ``structure``, e.g. a whole.pdb from ``gmx trjconv``, its positions are used instead: the
    topology gives the molecules, which such a PDB lacks, and the structure gives the positions.
    """
    bss = require_biosimspace()
    if structure is None:
        return bss.IO.readMolecules([str(topology), str(coordinates)])
    with tempfile.TemporaryDirectory() as tmp:
        positioned = Path(tmp) / Path(coordinates).name
        with_positions_of(Path(coordinates), Path(structure), positioned)
        return bss.IO.readMolecules([str(topology), str(positioned)])


def molecule_sizes(system) -> list[tuple[int, int]]:
    """(atoms, molecule index) of each molecule of a BioSimSpace System, largest first.

    ``makeFunnel()`` takes the largest molecule as the protein and the second largest as the ligand.
    """
    return sorted(((molecule.nAtoms(), index) for index, molecule in enumerate(system.getMolecules())),
                  reverse=True)


def nanometers(length) -> float:
    """A BioSimSpace ``Length`` in nm."""
    return length.nanometers().value()


def length_nm(value_nm: float):
    """A BioSimSpace ``Length`` of ``value_nm`` nm."""
    require_biosimspace()
    from BioSimSpace.Types import Length  # pyright: ignore[reportMissingImports]

    return Length(float(value_nm), "nanometer")


def atom_position_A(system, index: int, property_map: dict | None = None) -> np.ndarray:
    """Position (A) of system atom ``index``, via the public ``Atom.coordinates()``."""
    position = system.getAtom(int(index)).coordinates(property_map=property_map or {})
    return np.array([position.x().angstroms().value(), position.y().angstroms().value(),
                     position.z().angstroms().value()])
