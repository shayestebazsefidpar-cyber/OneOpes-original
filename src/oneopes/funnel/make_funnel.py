"""A BioSimSpace funnel: the ``makeFunnel()`` atom selections and the ``Funnel`` CV built on them."""

from dataclasses import dataclass
from typing import Any

from oneopes.funnel.bss import BssFunnel, BssMolecule, BssSystem, PropertyMap, require_biosimspace
from oneopes.funnel.geometry import FunnelParameters, funnel_parameters
from oneopes.funnel.p0_p1 import ALPHA_CARBON_NAME, FunnelAxis, funnel_axis


@dataclass(frozen=True)
class FunnelResult:
    """A BioSimSpace funnel: its axis from the makeFunnel selections, and the ``Funnel`` CV."""

    axis: FunnelAxis
    cv: BssFunnel
    """The BioSimSpace ``Funnel`` CV (e.g. ``cv.getCorrection()``)."""

    @property
    def parameters(self) -> FunnelParameters:
        """The CV's parameters as plain numbers."""
        return funnel_parameters(self.cv)


def make_funnel(
    system: BssSystem,
    protein: int | BssMolecule | None = None,
    ligand: int | BssMolecule | None = None,
    alpha_carbon_name: str = ALPHA_CARBON_NAME,
    property_map: PropertyMap | None = None,
    **funnel_kwargs: Any,
) -> FunnelResult:
    """Build a BioSimSpace funnel for a solvated protein-ligand ``system``.

    The named arguments go to ``makeFunnel()``, ``funnel_kwargs`` (e.g. ``width``, ``lower_bound``) to ``Funnel()``.
    """
    require_biosimspace()
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel  # pyright: ignore[reportMissingImports]

    axis = funnel_axis(system, protein=protein, ligand=ligand, alpha_carbon_name=alpha_carbon_name,
                       property_map=property_map)
    cv = Funnel(list(axis.atoms0), list(axis.atoms1), **funnel_kwargs)
    return FunnelResult(axis=axis, cv=cv)
