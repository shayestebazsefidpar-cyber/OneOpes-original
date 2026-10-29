"""Contact and Bridge occupancy of protein atoms around a bound ligand."""

from collections.abc import Sequence
from typing import Any

import numpy as np
import xarray as xr
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import distance_array

CONTACT_CUTOFF_A = 5.0
BRIDGE_WATER_CUTOFF_A = 3.5


def in_contact(candidates: np.ndarray, ligand: np.ndarray, box: np.ndarray | None) -> np.ndarray:
    """Per candidate, whether it is within CONTACT_CUTOFF_A of some ligand atom in this frame (positions in A)."""
    return distance_array(candidates, ligand, box=box).min(axis=1) <= CONTACT_CUTOFF_A


def bridged_by_water(candidates: np.ndarray, ligand: np.ndarray, waters: np.ndarray,
                     box: np.ndarray | None) -> np.ndarray:
    """Per candidate, whether some water oxygen is within BRIDGE_WATER_CUTOFF_A of both it and the ligand."""
    waters_at_ligand = waters[distance_array(waters, ligand, box=box).min(axis=1) <= BRIDGE_WATER_CUTOFF_A]
    return (distance_array(candidates, waters_at_ligand, box=box) <= BRIDGE_WATER_CUTOFF_A).any(axis=1)


class Contacts(AnalysisBase):
    """Fraction of frames each candidate atom is in contact with the ligand, and bridged to it by a water.

    Per frame (distances in A, minimum image): :func:`in_contact` and :func:`bridged_by_water`.

    Usage: ``results = Contacts(candidates, ligand, water_oxygens).run(start=..., stop=...).results``,
    then ``results.contact_occupancy`` and ``results.bridge_occupancy``, one value per candidate.
    """

    def __init__(self, candidates: AtomGroup, ligand: AtomGroup, water_oxygens: AtomGroup, **kwargs: Any) -> None:
        super().__init__(candidates.universe.trajectory, **kwargs)
        self.candidates = candidates
        self.ligand = ligand
        self.water_oxygens = water_oxygens

    def _prepare(self) -> None:
        self.results.contact_frames = np.zeros(self.candidates.n_atoms, dtype=np.int64)
        self.results.bridge_frames = np.zeros(self.candidates.n_atoms, dtype=np.int64)

    def _single_frame(self) -> None:
        box = self._ts.dimensions
        candidates, ligand = self.candidates.positions, self.ligand.positions
        self.results.contact_frames += in_contact(candidates, ligand, box)
        self.results.bridge_frames += bridged_by_water(candidates, ligand, self.water_oxygens.positions, box)

    def _conclude(self) -> None:
        self.results.contact_occupancy = self.results.contact_frames / self.n_frames
        self.results.bridge_occupancy = self.results.bridge_frames / self.n_frames


def measure_contacts(candidates: AtomGroup, ligand: AtomGroup, water_oxygens: AtomGroup, frames: range,
                     atoms: Sequence[str]) -> xr.Dataset:
    """``contact_occupancy`` and ``bridge_occupancy`` over ``frames``, along ``atom`` (named by ``atoms``)."""
    results = Contacts(candidates, ligand, water_oxygens).run(start=frames.start, stop=frames.stop).results
    return xr.Dataset({"contact_occupancy": ("atom", results.contact_occupancy),
                       "bridge_occupancy": ("atom", results.bridge_occupancy)},
                      coords={"atom": list(atoms)})
