"""
P0 / P1: the atoms BioSimSpace ``makeFunnel()`` selects, and the points derived from them.

``makeFunnel()`` (called unchanged; this package selects no atoms itself) returns two lists of
0-based system-level atom indices: ``atoms0``, alpha carbons deep in the protein (the funnel's
origin), and ``atoms1``, alpha carbons around the ligand (its inflection side). P0 and P1 are the
average positions of those atoms, in Angstrom.
"""

from dataclasses import dataclass

import numpy as np

from oneopes.errors import InputError
from oneopes.funnel.bss import BssMolecule, BssSystem, PropertyMap, atom_position_A, require_biosimspace

ALPHA_CARBON_NAME = "CA"
MIN_AXIS_LENGTH_A = 1e-6


@dataclass(frozen=True)
class P0P1Result:
    """The makeFunnel atom selections and the P0/P1 points derived from them (A)."""

    atoms0: list[int]
    """0-based system indices of the P0 (funnel origin) atoms."""
    atoms1: list[int]
    """0-based system indices of the P1 (inflection side) atoms."""
    p0: np.ndarray
    """Average position of ``atoms0``."""
    p1: np.ndarray
    """Average position of ``atoms1``."""

    def __post_init__(self) -> None:
        if not self.axis_length >= MIN_AXIS_LENGTH_A:
            raise InputError(f"P0 {self.p0} and P1 {self.p1} must be distinct points to define the funnel axis.")

    @property
    def axis_length(self) -> float:
        """|P1 - P0|."""
        return float(np.linalg.norm(self.p1 - self.p0))

    @property
    def axis(self) -> np.ndarray:
        """Unit vector P0 -> P1."""
        return (self.p1 - self.p0) / self.axis_length


def make_p0_p1(
    system: BssSystem,
    protein: int | BssMolecule | None = None,
    ligand: int | BssMolecule | None = None,
    alpha_carbon_name: str = ALPHA_CARBON_NAME,
    property_map: PropertyMap | None = None,
) -> P0P1Result:
    """atoms0/atoms1 from BioSimSpace ``makeFunnel()``, and P0/P1 as their average positions.

    All arguments are passed unchanged to
    ``makeFunnel(system, protein, ligand, alpha_carbon_name, property_map)``.
    """
    require_biosimspace()
    from BioSimSpace.Metadynamics.CollectiveVariable import makeFunnel  # pyright: ignore[reportMissingImports]

    property_map = property_map or {}
    atoms0, atoms1 = makeFunnel(
        system, protein=protein, ligand=ligand, alpha_carbon_name=alpha_carbon_name, property_map=property_map
    )
    positions0 = np.array([atom_position_A(system, i, property_map) for i in atoms0])
    positions1 = np.array([atom_position_A(system, i, property_map) for i in atoms1])
    return P0P1Result(atoms0=[int(i) for i in atoms0], atoms1=[int(i) for i in atoms1],
                      p0=positions0.mean(axis=0), p1=positions1.mean(axis=0))
