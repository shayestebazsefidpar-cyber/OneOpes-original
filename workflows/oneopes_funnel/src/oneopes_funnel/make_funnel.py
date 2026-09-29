"""
Top-level makeFunnel-type API around BioSimSpace.

``make_funnel(system, ...)`` = BioSimSpace ``makeFunnel()`` (atoms0/atoms1)
+ a BioSimSpace ``Funnel`` CV built from them, with P0/P1, axis, CV
parameters and funnel geometry exposed as plain numbers. Nothing here
changes the BioSimSpace science; see ``p0_p1.py`` and ``geometry.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import FunnelParameters, funnel_extent, funnel_parameters, funnel_wall_points
from .p0_p1 import ALPHA_CARBON_NAME, P0P1Result, make_p0_p1, require_biosimspace


@dataclass(frozen=True)
class FunnelResult:
    """A BioSimSpace funnel: makeFunnel atom selections, derived P0/P1, and the BSS ``Funnel`` CV.

    ``getCorrection()`` and every other CV method are available on ``.cv``.
    """

    p0p1: P0P1Result
    cv: object
    """The BioSimSpace ``Funnel`` collective variable (source of all CV quantities)."""
    parameters: FunnelParameters

    @property
    def atoms0(self) -> list[int]:
        return self.p0p1.atoms0

    @property
    def atoms1(self) -> list[int]:
        return self.p0p1.atoms1

    @property
    def p0(self) -> np.ndarray:
        return self.p0p1.p0

    @property
    def p1(self) -> np.ndarray:
        return self.p0p1.p1

    @property
    def axis(self) -> np.ndarray:
        return self.p0p1.axis

    @property
    def axis_length(self) -> float:
        return self.p0p1.axis_length

    def extent_nm(self, projection_nm) -> np.ndarray:
        """Funnel radius (nm) at projection(s) along the axis (nm) - ``cv.getExtent()``."""
        return funnel_extent(self.cv, projection_nm)

    def wall_points(self, radius: str = "cv", basis_ints: tuple[int, int] | None = None) -> np.ndarray:
        """viewFunnel-style wall points (A); see :func:`geometry.funnel_wall_points`."""
        return funnel_wall_points(self.p0, self.p1, self.cv, radius=radius, basis_ints=basis_ints)


def make_funnel(
    system,
    protein=None,
    ligand=None,
    alpha_carbon_name: str = ALPHA_CARBON_NAME,
    property_map: dict | None = None,
    **funnel_kwargs,
) -> FunnelResult:
    """Build a BioSimSpace funnel for ``system``.

    Parameters
    ----------
    system : BioSimSpace._SireWrappers.System
        Solvated protein-ligand system.
    protein, ligand, alpha_carbon_name, property_map
        Passed unchanged to BioSimSpace ``makeFunnel()``.
    **funnel_kwargs
        Passed unchanged to BioSimSpace ``Funnel(atoms0, atoms1, ...)``
        (``width``, ``buffer``, ``steepness``, ``inflection``, ``lower_bound``,
        ``upper_bound``, ``hill_width``, ``grid``) - BSS types and defaults.
    """
    require_biosimspace()
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    p0p1 = make_p0_p1(system, protein=protein, ligand=ligand, alpha_carbon_name=alpha_carbon_name, property_map=property_map)
    cv = Funnel(p0p1.atoms0, p0p1.atoms1, **funnel_kwargs)
    return FunnelResult(p0p1=p0p1, cv=cv, parameters=funnel_parameters(cv))
