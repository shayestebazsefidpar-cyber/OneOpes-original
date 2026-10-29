"""WaterFP, the hydration fingerprint of an atom (publication 10.1063/5.0287856, github.com/valeriorizzi/WaterFP).

For each atom, water oxygens are counted in thin spherical shells around it over the analysed frames
(:class:`WaterFP`), and :func:`hydration_from_counts` turns the counts into the water structure around the atom:
its water density n(r), the bulk density, g(r) = n(r) / bulk, and the fingerprint
FP = ∫ -2π·bulk·(g ln g - g + 1)·r² dr, which is 0 for bulk-like water and more negative the more
the water around the atom differs from bulk.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Annotated, Any

import numpy as np
import xarray as xr
from cyclopts import Parameter
from cyclopts.validators import Number
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import capped_distance
from MDAnalysis.units import convert
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
        return convert(self.edges_nm, "nm", "A")

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


@dataclass(frozen=True)
class HydrationOptions:
    """How the water around an atom is counted and normalised."""
    grid: Annotated[RadialGrid, Parameter(name="*")] = RadialGrid()
    norm_tail_bins: Annotated[int, Parameter(validator=Number(gte=1))] = NORM_TAIL_BINS_DEFAULT
    """Bins at the end of the profile whose mean is the bulk density."""

    def __post_init__(self) -> None:
        if not 1 <= self.norm_tail_bins <= self.grid.n_bins:
            raise InputError(f"norm_tail_bins must be between 1 and the {self.grid.n_bins} radial bins, "
                             f"got {self.norm_tail_bins}.")


def count_waters_by_distance(solute: np.ndarray, waters: np.ndarray, grid: RadialGrid,
                             box: np.ndarray | None) -> np.ndarray:
    """Water oxygens in each radial bin around each solute atom in one frame (positions in A)."""
    edges_a = grid.edges_a
    pairs, distances = capped_distance(solute, waters, max_cutoff=edges_a[-1], box=box)
    one_bin_per_atom = np.arange(len(solute) + 1)
    histogram, _, _ = np.histogram2d(pairs[:, 0], distances, bins=(one_bin_per_atom, edges_a))
    return histogram.astype(np.int64)


def hydration_from_counts(counts: np.ndarray, n_frames: int, atoms: Sequence[str],
                          options: HydrationOptions = HydrationOptions()) -> xr.Dataset:
    """Density n(r) (waters/nm^3), bulk density, g(r) and FP of each atom, from its counts summed over ``n_frames``.

    Raises InputError when an atom has no water in the bulk tail.
    """
    grid, tail_bins = options.grid, options.norm_tail_bins
    waters = xr.DataArray(counts, dims=("atom", "r_nm"), coords={"atom": list(atoms), "r_nm": grid.r_nm})
    shell_volumes = xr.DataArray(grid.shell_volumes_nm3, dims="r_nm", coords={"r_nm": grid.r_nm})
    density = waters / n_frames / shell_volumes
    bulk = density.isel(r_nm=slice(-tail_bins, None)).mean("r_nm")
    if (bulk == 0).any():
        no_water = bulk.atom[bulk == 0].values.tolist()
        raise InputError(f"No water in the bulk tail (last {tail_bins} bins, up to {grid.rmax_nm} nm) around "
                         f"atoms {no_water}: check the water selection and that the box holds water that far "
                         "from the solute.")
    g = density / bulk
    fp = (-2 * np.pi * bulk * (xlogy(g, g) - g + 1) * g.r_nm ** 2).integrate("r_nm")
    return xr.Dataset({"density": density, "bulk": bulk, "g": g, "fp": fp})


class WaterFP(AnalysisBase):
    """Water oxygens per radial bin around each solute atom; see ``results.counts`` and ``results.hydration``.

    Counts add up across frames, so ``run(backend="multiprocessing")`` works.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls) -> tuple[str, ...]:  # pyright: ignore[reportIncompatibleMethodOverride]
        return ("serial", "multiprocessing", "dask")

    def __init__(self, solute: AtomGroup, water_oxygens: AtomGroup, *,
                 options: HydrationOptions = HydrationOptions(), labels: Sequence[str] | None = None,
                 **kwargs: Any) -> None:
        if labels is not None and len(labels) != len(solute):
            raise ValueError(f"{len(labels)} labels for {len(solute)} solute atoms.")
        super().__init__(solute.universe.trajectory, **kwargs)
        self.solute = solute
        self.water_oxygens = water_oxygens
        self.options = options
        self.labels = list(solute.names) if labels is None else list(labels)

    def _prepare(self) -> None:
        self.results.counts = np.zeros((self.solute.n_atoms, self.options.grid.n_bins), dtype=np.int64)

    def _single_frame(self) -> None:
        """Add this frame's counts; warn when the box is too small for rmax."""
        box, grid = self._ts.dimensions, self.options.grid
        if box is not None and grid.edges_a[-1] > 0.5 * min(box[:3]):
            warnings.warn(
                f"rmax ({grid.rmax_nm} nm) is larger than half the shortest box edge: with minimum-image "
                "distances the outer shells are only partly sampled, so the bulk density and every FP "
                "are biased. Use a larger box or a smaller rmax.",
                RuntimeWarning,
                stacklevel=2,
            )
        self.results.counts += count_waters_by_distance(self.solute.positions, self.water_oxygens.positions,
                                                        grid, box)

    def _get_aggregator(self) -> ResultsGroup:
        return ResultsGroup(lookup={"counts": ResultsGroup.ndarray_sum})

    def _conclude(self) -> None:
        measured = hydration_from_counts(self.results.counts, self.n_frames, self.labels, self.options)
        self.results.hydration = measured.assign_coords(atom_index=("atom", self.solute.indices))


def measure_hydration(solute: AtomGroup, water_oxygens: AtomGroup, frames: range,
                      options: HydrationOptions = HydrationOptions(),
                      labels: Sequence[str] | None = None) -> xr.Dataset:
    """The hydration of each ``solute`` atom over ``frames`` of its trajectory."""
    analysis = WaterFP(solute, water_oxygens, options=options, labels=labels)
    return analysis.run(start=frames.start, stop=frames.stop).results.hydration
