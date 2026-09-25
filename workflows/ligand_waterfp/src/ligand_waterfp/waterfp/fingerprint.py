"""
The WaterFP method (github.com/valeriorizzi/WaterFP), reimplemented with
MDAnalysis: each solute atom's water-oxygen radial density profile n(r)
over a trajectory frame range, and the hydration fingerprint derived
from it,

    FP = integral_0^rmax  -2*pi*norm * (g ln(g) - g + 1) * r^2 dr

where g(r) = n(r)/norm and norm is the empirical bulk density (the mean
of the last `norm_tail_bins` bins of n(r)). The integrand is a water
excess-entropy density relative to bulk: FP grows wherever a solute atom
makes the surrounding water systematically denser or sparser than bulk.

Defaults match the reference scripts (calc_rdf.sh: 2001 bins of 0.001 nm
up to rmax = 2.001 nm; fp.py: 500-bin bulk tail).
"""

from typing import NamedTuple

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array
from scipy.integrate import trapezoid
from scipy.special import xlogy

RDF_RMAX_NM_DEFAULT = 2.001
RDF_BINWIDTH_NM_DEFAULT = 0.001
NORM_TAIL_BINS_DEFAULT = 500


class RadialBins(NamedTuple):
    edges_nm: np.ndarray
    centers_nm: np.ndarray
    edges_angstrom: np.ndarray
    shell_volumes_nm3: np.ndarray


def make_bins(
    rmax_nm: float = RDF_RMAX_NM_DEFAULT, binwidth_nm: float = RDF_BINWIDTH_NM_DEFAULT
) -> RadialBins:
    """The radial grid for the density profile: bin edges/centers in nm,
    the same edges in Angstrom (MDAnalysis distances), and the spherical
    shell volume of each bin in nm^3."""
    nbins = int(round(rmax_nm / binwidth_nm))
    edges_nm = np.linspace(0.0, rmax_nm, nbins + 1)
    centers_nm = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    edges_angstrom = edges_nm * 10.0
    shell_volumes_nm3 = (4.0 / 3.0) * np.pi * (edges_nm[1:] ** 3 - edges_nm[:-1] ** 3)
    return RadialBins(edges_nm, centers_nm, edges_angstrom, shell_volumes_nm3)


def _water_counts_per_shell(
    solute: mda.AtomGroup,
    water: mda.AtomGroup,
    edges_angstrom: np.ndarray,
    box: np.ndarray,
) -> np.ndarray:
    """One frame's histogram of solute-water distances, per solute atom:
    shape (len(solute), len(edges_angstrom) - 1)."""
    distances = distance_array(solute.positions, water.positions, box=box)  # Angstrom
    return np.array([np.histogram(d, bins=edges_angstrom)[0] for d in distances])


def compute_density_profile(
    solute: mda.AtomGroup,
    water: mda.AtomGroup,
    start_frame: int,
    end_frame: int,
    bins: RadialBins,
) -> np.ndarray:
    """Per-atom raw water number-density profile n(r) (waters/nm^3),
    averaged over all frames in [start_frame, end_frame) of the trajectory
    both AtomGroups belong to.

    Returns an array of shape (len(solute), n_bins): one row per solute
    atom, one column per radial bin of `bins`.
    """
    trajectory = solute.universe.trajectory
    if not 0 <= start_frame < end_frame <= len(trajectory):
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for a trajectory "
            f"of {len(trajectory)} frames"
        )

    counts = np.zeros((len(solute), len(bins.centers_nm)))
    for ts in trajectory[start_frame:end_frame]:
        counts += _water_counts_per_shell(
            solute, water, bins.edges_angstrom, ts.dimensions
        )

    mean_counts = counts / (end_frame - start_frame)  # per shell, per frame
    return mean_counts / bins.shell_volumes_nm3  # -> number density, waters/nm^3


class FingerprintResult(NamedTuple):
    fp: float
    g_r: np.ndarray
    norm: float


def fp_from_density_profile(
    n_r: np.ndarray,
    r_centers_nm: np.ndarray,
    norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT,
) -> FingerprintResult:
    """FP of one atom's raw number-density profile n_r (waters/nm^3 per
    bin) on the radial grid r_centers_nm.

    xlogy(0, 0) = 0, so empty bins get the integrand's well-defined
    g -> 0 limit (g ln g - g + 1 -> 1) with no special-casing."""
    norm = float(np.mean(n_r[-norm_tail_bins:]))
    if norm <= 0:
        raise ValueError(
            f"Bulk water density is 0 (mean of the last {norm_tail_bins} bins of "
            "n(r)) - the profile never reaches bulk water. Likely causes: rmax too "
            "small for this system, too few frames, or a buried/misselected atom."
        )
    g = n_r / norm
    entropy_term = xlogy(g, g) - g + 1.0  # g ln(g) - g + 1
    integrand = -2.0 * np.pi * norm * entropy_term * r_centers_nm**2
    fp = float(trapezoid(integrand, x=r_centers_nm))
    return FingerprintResult(fp, g, norm)
