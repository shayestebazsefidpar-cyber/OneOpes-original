"""
``make_funnel(system, ...)``: the BioSimSpace ``makeFunnel()`` atom selections with the P0/P1 derived
from them (``p0_p1.py``), and the BioSimSpace ``Funnel`` CV built on them with its parameters
(``geometry.py``).
"""

from dataclasses import dataclass

from oneopes.funnel.bss import require_biosimspace
from oneopes.funnel.geometry import FunnelParameters, funnel_parameters
from oneopes.funnel.p0_p1 import ALPHA_CARBON_NAME, P0P1Result, make_p0_p1


@dataclass(frozen=True)
class FunnelResult:
    """A BioSimSpace funnel: the makeFunnel selections with P0/P1, and the ``Funnel`` CV."""

    p0p1: P0P1Result
    cv: object
    """The BioSimSpace ``Funnel`` CV (e.g. ``cv.getCorrection()``)."""

    @property
    def parameters(self) -> FunnelParameters:
        """The CV's parameters as plain numbers (see :func:`funnel_parameters`)."""
        return funnel_parameters(self.cv)


def make_funnel(
    system,
    protein=None,
    ligand=None,
    alpha_carbon_name: str = ALPHA_CARBON_NAME,
    property_map: dict | None = None,
    **funnel_kwargs,
) -> FunnelResult:
    """Build a BioSimSpace funnel for ``system``, a solvated protein-ligand BioSimSpace System.

    ``protein``, ``ligand``, ``alpha_carbon_name`` and ``property_map`` go unchanged to BioSimSpace
    ``makeFunnel()``; ``funnel_kwargs`` (``width``, ``buffer``, ``steepness``, ``inflection``,
    ``lower_bound``, ``upper_bound``, ...) go unchanged to BioSimSpace ``Funnel(atoms0, atoms1, ...)``.
    """
    require_biosimspace()
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel  # pyright: ignore[reportMissingImports]

    p0p1 = make_p0_p1(system, protein=protein, ligand=ligand, alpha_carbon_name=alpha_carbon_name,
                      property_map=property_map)
    cv = Funnel(p0p1.atoms0, p0p1.atoms1, **funnel_kwargs)
    return FunnelResult(p0p1=p0p1, cv=cv)
