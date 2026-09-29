"""
P0 / P1 (atoms0 / atoms1) selection and derived P0/P1 coordinates.

Source of truth: BioSimSpace ``makeFunnel()`` (BioSimSpace 2024.4.1,
``BioSimSpace/Metadynamics/CollectiveVariable/_funnel.py`` lines 711-989;
GPL-3.0-or-later, (c) 2017-2025 Lester Hedges; adapted there from
``funnel_maker.py`` by Dominykas Lukauskis).

:func:`make_p0_p1` calls BioSimSpace ``makeFunnel()`` unchanged and takes the
atom selections it returns:

* ``atoms1`` - every alpha carbon within 10 A of the ligand COM.
* ``atoms0`` - every alpha carbon within 7 A of a point 10 A from the atoms1
  centroid, on the side away from the open (solvent) direction.

``makeFunnel()`` returns only these atom selections, not coordinates. P0/P1
here are **derived** from them: the plain average position of the ``atoms0``
/ ``atoms1`` atoms (the same formula makeFunnel applies internally to atoms1,
lines 949-954). Because every selected atom is a CA, this equals the
mass-weighted COM that PLUMED computes from the same atoms.

Conventions
-----------
* Coordinates and lengths are in Angstrom.
* ``atoms0`` / ``atoms1`` are 0-based system-level atom indices, exactly as
  returned by makeFunnel (``System.getIndex``). BioSimSpace's PLUMED writer
  writes them as 1-based serials ``index + 1`` (``p1``/``p2`` COMs). BSS
  ``viewFunnel()`` instead looks them up in the protein's own atom list; that
  convention is NOT used here.
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
    """makeFunnel atoms0: 0-based system indices of the P0 (funnel origin) CA atoms."""
    atoms1: list[int]
    """makeFunnel atoms1: 0-based system indices of the P1 (inflection side) CA atoms."""
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


def _xyz(v) -> tuple[float, float, float]:
    """Sire vector -> floats in A (components may carry Sire length units)."""
    return tuple(c.value() if hasattr(c, "value") else float(c) for c in (v.x(), v.y(), v.z()))


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
    coord_prop = property_map.get("coordinates", "coordinates")
    coords = np.zeros((max([*atoms0, *atoms1]) + 1, 3))
    for i in set(atoms0) | set(atoms1):
        coords[i] = _xyz(system.getAtom(i)._sire_object.property(coord_prop))
    return p0_p1_from_atoms(atoms0, atoms1, coords)
