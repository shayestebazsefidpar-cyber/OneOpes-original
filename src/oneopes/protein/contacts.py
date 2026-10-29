"""Contact and Bridge occupancy of protein atoms around a bound ligand."""

from collections.abc import Sequence

import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import distance_array

CONTACT_CUTOFF_A = 5.0
BRIDGE_WATER_CUTOFF_A = 3.5


class Contacts(AnalysisBase):
    """Fraction of frames each candidate atom is in contact with the ligand, and bridged to it by a water.

    Per frame (distances in A, minimum image):
    * Contact: the candidate is within 5 A of some ligand heavy atom.
    * Bridge: some water oxygen is within 3.5 A of the candidate and within 3.5 A of some ligand heavy atom.

    Usage: ``results = Contacts(candidates, ligand, water_oxygens).run(start=..., stop=...).results``,
    then ``results.contact_occupancy`` and ``results.bridge_occupancy``, one value per candidate.
    """

    def __init__(self, candidates: AtomGroup, ligand: AtomGroup, water_oxygens: AtomGroup, **kwargs) -> None:
        super().__init__(candidates.universe.trajectory, **kwargs)
        self.candidates = candidates
        self.ligand = ligand
        self.water_oxygens = water_oxygens

    def _prepare(self) -> None:
        self.results.contact_frames = np.zeros(self.candidates.n_atoms, dtype=np.int64)
        self.results.bridge_frames = np.zeros(self.candidates.n_atoms, dtype=np.int64)

    def _single_frame(self) -> None:
        box = self._ts.dimensions
        candidates, ligand, water = self.candidates.positions, self.ligand.positions, self.water_oxygens.positions
        self.results.contact_frames += distance_array(candidates, ligand, box=box).min(axis=1) <= CONTACT_CUTOFF_A
        water_near_ligand = water[distance_array(water, ligand, box=box).min(axis=1) <= BRIDGE_WATER_CUTOFF_A]
        bridged = distance_array(candidates, water_near_ligand, box=box) <= BRIDGE_WATER_CUTOFF_A
        self.results.bridge_frames += bridged.any(axis=1)

    def _conclude(self) -> None:
        self.results.contact_occupancy = self.results.contact_frames / self.n_frames
        self.results.bridge_occupancy = self.results.bridge_frames / self.n_frames

    def table(self, labels: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per candidate: atom, Contact_occupancy_5A, Bridge_occupancy, n_frames.

        ``labels`` name the atoms (default: candidate atom names).
        """
        return pd.DataFrame({
            "atom": self.candidates.names if labels is None else labels,
            "Contact_occupancy_5A": self.results.contact_occupancy,
            "Bridge_occupancy": self.results.bridge_occupancy,
            "n_frames": self.n_frames,
        })
