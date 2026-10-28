"""Functionality to compute WaterFP as in publication 10.1063/5.0287856

Original at: github.com/valeriorizzi/WaterFP
"""

from dataclasses import dataclass
import numpy as np
from MDAnalysis.analysis.base import AnalysisBase
from MDAnalysis.analysis.results import ResultsGroup
from MDAnalysis.core.groups import AtomGroup
from MDAnalysis.lib.distances import capped_distance, distance_array
from scipy.special import xlogy


RDF_RMAX_NM_DEFAULT = 2.001
RDF_BINWIDTH_NM_DEFAULT = 0.001
NORM_TAIL_BINS_DEFAULT = 500

@dataclass(frozen=True, eq=False)
class WaterCounts:
    counts: np.ndarray          # (n_atoms, n_bins) int64: water O counts per bin, summed over frames
    n_frames: int               # frames contributing to counts
    solute_indices: np.ndarray  # (n_atoms,) global atom indices; row i of counts <-> solute_indices[i]
    rmax_nm: float
    binwidth_nm: float

    @property
    def n_bins(self) -> int:
        return round(self.rmax_nm / self.binwidth_nm)

    @property
    def edges_nm(self) -> np.ndarray:
        return np.linspace(0.0, self.rmax_nm, self.n_bins + 1)

    @property
    def r_nm(self) -> np.ndarray:            # bin centres
        e = self.edges_nm
        return 0.5 * (e[:-1] + e[1:])

    @property
    def shell_volumes_nm3(self) -> np.ndarray:
        e = self.edges_nm
        return (4.0 / 3.0) * np.pi * (e[1:] ** 3 - e[:-1] ** 3)

    @property
    def density_nm3(self) -> np.ndarray:     # (n_atoms, n_bins)
        return self.counts / self.n_frames / self.shell_volumes_nm3


def fingerprint(counts: WaterCounts,
                norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT) -> tuple[np.ndarray, np.ndarray]:
    """WaterFP and g(r) of every solute atom in ``counts``.

    g(r) is the water density relative to the bulk plateau (mean of the last
    ``norm_tail_bins`` bins); FP integrates -2*pi*bulk*(g ln g - g + 1)*r^2 over r.
    Returns (fp of shape (n_atoms,), g of shape (n_atoms, n_bins)).
    """
    density = counts.density_nm3
    bulk = density[:, -norm_tail_bins:].mean(axis=1, keepdims=True)
    g = density / bulk
    integrand = -2 * np.pi * bulk * (xlogy(g, g) - g + 1) * counts.r_nm ** 2  # xlogy(0, 0) = 0
    fp = np.trapezoid(integrand, dx=counts.binwidth_nm, axis=1)
    return fp, g


class WaterFP(AnalysisBase):
    """WaterFP of each solute atom over the analysed frames.

    Usage: ``results = WaterFP(solute, water_oxygens).run(start=..., stop=...).results``, then
    ``results.fp`` (n_atoms,), ``results.g`` (n_atoms, n_bins), ``results.r_nm`` (n_bins,) and
    ``results.water_counts`` (the raw, additive WaterCounts, e.g. for combining blocks).
    Counts add up across frames, so the analysis can also run in parallel,
    e.g. ``run(backend="multiprocessing", n_workers=4)``.
    """

    _analysis_algorithm_is_parallelizable = True

    @classmethod
    def get_supported_backends(cls) -> tuple[str, ...]:
        return ("serial", "multiprocessing", "dask")

    def __init__(self, solute: AtomGroup, water_oxygens: AtomGroup, *,
                 rmax_nm: float = RDF_RMAX_NM_DEFAULT, binwidth_nm: float = RDF_BINWIDTH_NM_DEFAULT,
                 norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT, **kwargs) -> None:
        super().__init__(solute.universe.trajectory, **kwargs)
        self.solute = solute
        self.water_oxygens = water_oxygens
        self.rmax_nm = rmax_nm
        self.binwidth_nm = binwidth_nm
        self.norm_tail_bins = norm_tail_bins
        n_bins = round(rmax_nm / binwidth_nm)
        self._distance_edges_a = 10.0 * np.linspace(0.0, rmax_nm, n_bins + 1)  # MDAnalysis uses Angstrom
        self._atom_edges = np.arange(len(solute) + 1)  # one histogram row per solute atom

    def _prepare(self) -> None:
        shape = (len(self._atom_edges) - 1, len(self._distance_edges_a) - 1)
        self.results.counts = np.zeros(shape, dtype=np.int64)

    def _single_frame(self) -> None:
        pairs, distances = capped_distance(self.solute.positions, self.water_oxygens.positions,
                                           max_cutoff=self._distance_edges_a[-1], box=self._ts.dimensions)
        histogram, _, _ = np.histogram2d(pairs[:, 0], distances, bins=(self._atom_edges, self._distance_edges_a))
        self.results.counts += histogram.astype(np.int64)

    def _get_aggregator(self) -> ResultsGroup:
        return ResultsGroup(lookup={"counts": ResultsGroup.ndarray_sum})

    def _conclude(self) -> None:
        water_counts = WaterCounts(counts=self.results.counts, n_frames=self.n_frames,
                                   solute_indices=self.solute.indices.copy(),
                                   rmax_nm=self.rmax_nm, binwidth_nm=self.binwidth_nm)
        self.results.water_counts = water_counts
        self.results.fp, self.results.g = fingerprint(water_counts, self.norm_tail_bins)
        self.results.r_nm = water_counts.r_nm


def make_bins(rmax_nm=RDF_RMAX_NM_DEFAULT, binwidth_nm=RDF_BINWIDTH_NM_DEFAULT):
    """Return (edges_nm, centers_nm, edges_angstrom, shell_volumes_nm3)."""
    nbins = int(round(rmax_nm / binwidth_nm))
    edges_nm = np.linspace(0.0, rmax_nm, nbins + 1)
    centers_nm = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    edges_a = edges_nm * 10.0  # MDAnalysis distances are in Angstrom
    shell_vol_nm3 = (4.0 / 3.0) * np.pi * (edges_nm[1:] ** 3 - edges_nm[:-1] ** 3)
    return edges_nm, centers_nm, edges_a, shell_vol_nm3

def compute_density_profile(u, solute_indices, water_indices, start_frame, end_frame,
                             edges_a, shell_vol_nm3):
    """Per-atom raw water number-density histogram n(r) (waters/nm^3),
    averaged over all frames in [start_frame, end_frame) of trajectory `u`.

    Returns an array of shape (len(solute_indices), len(edges_a) - 1).
    """
    n_solute = len(solute_indices)
    n_bins = len(edges_a) - 1
    hist_sum = np.zeros((n_solute, n_bins))
    n_frames = end_frame - start_frame
    for fi in range(start_frame, end_frame):
        u.trajectory[fi]
        solute_pos = u.atoms.positions[solute_indices]
        water_pos = u.atoms.positions[water_indices]
        d = distance_array(solute_pos, water_pos, box=u.dimensions)  # Angstrom
        for ai in range(n_solute):
            h, _ = np.histogram(d[ai], bins=edges_a)
            hist_sum[ai] += h
    mean_hist = hist_sum / n_frames        # mean neighbour count per bin per frame
    n_r = mean_hist / shell_vol_nm3        # number density, waters/nm^3
    return n_r


def fp_from_density_profile(n_r, r_centers_nm, binwidth_nm, norm_tail_bins=NORM_TAIL_BINS_DEFAULT):
    """Given one atom's raw number-density profile n_r (waters/nm^3 per
    bin) and its radial bin centers, return (FP, g(r), norm)."""
    norm = float(np.mean(n_r[-norm_tail_bins:]))
    g = n_r / norm
    r = r_centers_nm
    integrand = np.empty_like(g)
    mask0 = (g == 0)
    integrand[mask0] = -2 * np.pi * norm * (r[mask0] ** 2)  # lim g->0 of (g ln g - g + 1) = 1
    gnz = g[~mask0]
    integrand[~mask0] = -2 * np.pi * norm * (gnz * np.log(gnz) - gnz + 1) * (r[~mask0] ** 2)
    fp = float(np.trapezoid(integrand, dx=binwidth_nm))
    return fp, g, norm
