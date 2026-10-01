"""
P0 / P1 (atoms0 / atoms1) selection and derived P0/P1 coordinates.

The atom selections come from the public BioSimSpace function
``BioSimSpace.Metadynamics.CollectiveVariable.makeFunnel()``, called
unchanged; this package does not select atoms itself.

``makeFunnel()`` returns two lists of atom indices:

* ``atoms1`` - alpha carbons around the ligand (the funnel's inflection side).
* ``atoms0`` - alpha carbons deeper in the protein (the funnel's origin).

It returns no coordinates. P0/P1 here are **derived** from the selections:
the plain average position of the ``atoms0`` / ``atoms1`` atoms. Every
selected atom is an alpha carbon, so this equals their mass-weighted centre.

Conventions
-----------
* Coordinates and lengths are in Angstrom.
* ``atoms0`` / ``atoms1`` are 0-based system-level atom indices, exactly as
  returned by ``makeFunnel()`` (``System.getAtom(i)`` / ``System.getIndex``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

ALPHA_CARBON_NAME = "CA"


def require_biosimspace():
    """Import BioSimSpace or fail with an actionable message."""
    try:
        import BioSimSpace
    except ImportError as err:
        raise ImportError(
            "oneopes_funnel needs BioSimSpace (conda-only, e.g. the 'biosimspace' "
            "conda env); it is not pip-installable."
        ) from err
    return BioSimSpace


@dataclass(frozen=True)
class P0P1Result:
    """BioSimSpace atom selections and the P0/P1 coordinates derived from them (A)."""

    atoms0: list[int]
    """makeFunnel atoms0: 0-based system indices of the P0 (funnel origin) atoms."""
    atoms1: list[int]
    """makeFunnel atoms1: 0-based system indices of the P1 (inflection side) atoms."""
    p0: np.ndarray
    """Derived: average position of ``atoms0``."""
    p1: np.ndarray
    """Derived: average position of ``atoms1``."""
    axis: np.ndarray
    """Derived: unit vector P0 -> P1."""
    axis_length: float
    """Derived: |P1 - P0|."""


def centroid(coords: np.ndarray) -> np.ndarray:
    """Unweighted mean position of an (N, 3) array."""
    coords = np.asarray(coords, dtype=float)
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError("centroid() needs a non-empty (N, 3) array.")
    return coords.mean(axis=0)


def p0_p1_from_atoms(atoms0: Sequence[int], atoms1: Sequence[int], coords: np.ndarray) -> P0P1Result:
    """Derive P0/P1/axis from atom selections; ``coords`` is indexed by system atom index (A)."""
    coords = np.asarray(coords, dtype=float)
    p0 = centroid(coords[list(atoms0)])
    p1 = centroid(coords[list(atoms1)])
    axis_vec = p1 - p0
    axis_length = float(np.linalg.norm(axis_vec))
    return P0P1Result(
        atoms0=[int(i) for i in atoms0],
        atoms1=[int(i) for i in atoms1],
        p0=p0,
        p1=p1,
        axis=axis_vec / axis_length,
        axis_length=axis_length,
    )


def atom_coordinates_A(system, index: int, property_map: dict | None = None) -> np.ndarray:
    """Coordinates (A) of system atom ``index`` via the public ``Atom.coordinates()``."""
    c = system.getAtom(int(index)).coordinates(property_map=property_map or {})
    return np.array([c.x().angstroms().value(), c.y().angstroms().value(), c.z().angstroms().value()])


def make_p0_p1(
    system,
    protein=None,
    ligand=None,
    alpha_carbon_name: str = ALPHA_CARBON_NAME,
    property_map: dict | None = None,
) -> P0P1Result:
    """atoms0/atoms1 from BioSimSpace ``makeFunnel()``, plus derived P0/P1.

    All arguments are passed unchanged to
    ``makeFunnel(system, protein, ligand, alpha_carbon_name, property_map)``.
    """
    require_biosimspace()
    from BioSimSpace.Metadynamics.CollectiveVariable import makeFunnel

    property_map = property_map or {}
    atoms0, atoms1 = makeFunnel(
        system, protein=protein, ligand=ligand, alpha_carbon_name=alpha_carbon_name, property_map=property_map
    )
    coords = np.zeros((max([*atoms0, *atoms1]) + 1, 3))
    for i in set(atoms0) | set(atoms1):
        coords[i] = atom_coordinates_A(system, i, property_map)
    return p0_p1_from_atoms(atoms0, atoms1, coords)
