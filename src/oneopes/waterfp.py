"""Functionality to compute WaterFP as in publication 10.1063/5.0287856

Original at: github.com/valeriorizzi/WaterFP
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
import numpy as np
import pandas as pd
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import capped_distance
from scipy.special import xlogy

from oneopes.errors import InputError

NORM_TAIL_BINS_DEFAULT = 500


@dataclass(frozen=True)
class RadialGrid:
    """Uniform radial bins from 0 to rmax_nm (the WaterFP reference: 2.001 nm in 0.001 nm bins)."""

    rmax_nm: float = 2.001
    """Largest water-O distance counted (nm)."""
    binwidth_nm: float = 0.001
    """Width of one radial bin (nm)."""

    def __post_init__(self) -> None:
        if not 0 < self.binwidth_nm <= self.rmax_nm:
            raise InputError(f"The radial grid needs 0 < binwidth <= rmax, got binwidth {self.binwidth_nm} nm "
                             f"and rmax {self.rmax_nm} nm.")

    @property
    def n_bins(self) -> int:
        return round(self.rmax_nm / self.binwidth_nm)

    @property
    def edges_nm(self) -> np.ndarray:
        return np.linspace(0.0, self.rmax_nm, self.n_bins + 1)

    @property
    def edges_a(self) -> np.ndarray:
        """The bin edges in Angstrom, the unit of MDAnalysis distances."""
        return 10.0 * self.edges_nm

    @property
    def r_nm(self) -> np.ndarray:
        """Centre of each bin."""
        lower, upper = self.edges_nm[:-1], self.edges_nm[1:]
        return 0.5 * (lower + upper)

    @property
    def shell_volumes_nm3(self) -> np.ndarray:
        """Volume of the spherical shell each bin covers around an atom."""
        lower, upper = self.edges_nm[:-1], self.edges_nm[1:]
        return (4.0 / 3.0) * np.pi * (upper ** 3 - lower ** 3)


@dataclass(frozen=True, eq=False)
class WaterCounts:
    """Water-oxygen counts per radial bin around each solute atom, summed over frames."""
    counts: np.ndarray
    """(n_atoms, grid.n_bins) int64."""
    n_frames: int
    """Frames contributing to ``counts``."""
    solute_indices: np.ndarray
    """(n_atoms,) atom indices in the universe; row i of ``counts`` is atom ``solute_indices[i]``."""
    grid: RadialGrid

    @property
    def density_nm3(self) -> np.ndarray:
        """Water-O number density n(r) around each atom (waters/nm^3), shape (n_atoms, n_bins)."""
        return self.counts / self.n_frames / self.grid.shell_volumes_nm3


def fingerprint(counts: WaterCounts,
                norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """WaterFP and g(r) of every solute atom in ``counts``.

    g(r) is the water density relative to the bulk plateau (mean of the last
    ``norm_tail_bins`` bins); FP integrates -2*pi*bulk*(g ln g - g + 1)*r^2 over r.
    Returns (fp of shape (n_atoms,), g of shape (n_atoms, n_bins)).
    Raises ValueError when an atom has no water at all in the bulk tail.
    """
    density = counts.density_nm3
    bulk = density[:, -norm_tail_bins:].mean(axis=1, keepdims=True)
    no_bulk_water = bulk[:, 0] == 0
    if no_bulk_water.any():
        raise ValueError(
            f"No water in the bulk tail (last {norm_tail_bins} bins, up to {counts.grid.rmax_nm} nm) around "
            f"solute atoms {counts.solute_indices[no_bulk_water].tolist()}: check the water selection "
            "and that the box holds water that far from the solute."
        )
    g = density / bulk
    integrand = -2 * np.pi * bulk * (xlogy(g, g) - g + 1) * counts.grid.r_nm ** 2  # xlogy(0, 0) = 0
    fp = np.asarray(np.trapezoid(integrand, dx=counts.grid.binwidth_nm, axis=1))
    return fp, g


class WaterFP(AnalysisBase):
    """WaterFP of each solute atom over the analysed frames.

    Usage: ``results = WaterFP(solute, water_oxygens).run(start=..., stop=...).results``, then
    ``results.fp`` (n_atoms,), ``results.g`` (n_atoms, n_bins), ``results.r_nm`` (n_bins,) and
    ``results.water_counts`` (the raw, additive WaterCounts, e.g. for combining blocks).
    After ``run()``, ``fp_table()`` and ``profile_table()`` give the same results as tables.
    Counts add up across frames, so the analysis can also run in parallel,
    e.g. ``run(backend="multiprocessing", n_workers=4)``.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return ("serial", "multiprocessing", "dask")

    def __init__(self, solute: AtomGroup, water_oxygens: AtomGroup, *, grid: RadialGrid = RadialGrid(),
                 norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT, **kwargs) -> None:
        if not 1 <= norm_tail_bins <= grid.n_bins:
            raise InputError(
                f"norm_tail_bins must be between 1 and the {grid.n_bins} radial bins, got {norm_tail_bins}.")
        super().__init__(solute.universe.trajectory, **kwargs)
        self.solute = solute
        self.water_oxygens = water_oxygens
        self.grid = grid
        self.norm_tail_bins = norm_tail_bins

    def _prepare(self) -> None:
        self.results.counts = np.zeros((self.solute.n_atoms, self.grid.n_bins), dtype=np.int64)

    def _single_frame(self) -> None:
        edges_a = self.grid.edges_a
        box = self._ts.dimensions
        if box is not None and edges_a[-1] > 0.5 * min(box[:3]):
            # constant message: Python shows it once, not once per frame
            warnings.warn(
                f"rmax ({self.grid.rmax_nm} nm) is larger than half the shortest box edge: with minimum-image "
                "distances the outer shells are only partly sampled, so the bulk density and every FP "
                "are biased. Use a larger box or a smaller rmax.",
                RuntimeWarning,
                stacklevel=2,
            )
        pairs, distances = capped_distance(self.solute.positions, self.water_oxygens.positions,
                                           max_cutoff=edges_a[-1], box=box)
        one_bin_per_atom = np.arange(self.solute.n_atoms + 1)
        histogram, _, _ = np.histogram2d(pairs[:, 0], distances, bins=(one_bin_per_atom, edges_a))
        self.results.counts += histogram.astype(np.int64)

    def _get_aggregator(self) -> ResultsGroup:
        return ResultsGroup(lookup={"counts": ResultsGroup.ndarray_sum})

    def _conclude(self) -> None:
        water_counts = WaterCounts(counts=self.results.counts, n_frames=self.n_frames,
                                   solute_indices=self.solute.indices.copy(), grid=self.grid)
        self.results.water_counts = water_counts
        self.results.fp, self.results.g = fingerprint(water_counts, self.norm_tail_bins)
        self.results.r_nm = self.grid.r_nm

    def fp_table(self, labels: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per solute atom: atom, FP. ``labels`` name the atoms (default: solute atom names)."""
        return pd.DataFrame({"atom": self._atom_labels(labels), "FP": self.results.fp})

    def profile_table(self, labels: Sequence[str] | None = None) -> pd.DataFrame:
        """One row per (solute atom, r): atom, r_nm, n_r (water-O density, waters/nm^3), g_r.

        ``labels`` name the atoms (default: solute atom names).
        """
        labels = self._atom_labels(labels)
        r_nm = self.results.r_nm
        return pd.DataFrame({
            "atom": np.repeat(labels, len(r_nm)),
            "r_nm": np.tile(r_nm, len(labels)),
            "n_r": self.results.water_counts.density_nm3.ravel(),  # row by row: one atom after the other
            "g_r": self.results.g.ravel(),
        })

    def _atom_labels(self, labels: Sequence[str] | None) -> list[str]:
        if labels is None:
            return list(self.solute.names)
        if len(labels) != len(self.solute):
            raise ValueError(f"{len(labels)} labels for {len(self.solute)} solute atoms.")
        return list(labels)
