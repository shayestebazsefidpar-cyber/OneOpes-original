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

import MDAnalysis as mda
import numpy as np
import pandas as pd
from MDAnalysis.lib.distances import distance_array
from scipy.integrate import trapezoid
from scipy.special import xlogy

RDF_RMAX_NM_DEFAULT = 2.001
RDF_BINWIDTH_NM_DEFAULT = 0.001
NORM_TAIL_BINS_DEFAULT = 500


def make_bins(rmax_nm=RDF_RMAX_NM_DEFAULT, binwidth_nm=RDF_BINWIDTH_NM_DEFAULT):
    """Return (edges_nm, centers_nm, edges_angstrom, shell_volumes_nm3)."""
    nbins = int(round(rmax_nm / binwidth_nm))
    edges_nm = np.linspace(0.0, rmax_nm, nbins + 1)
    centers_nm = 0.5 * (edges_nm[:-1] + edges_nm[1:])
    edges_a = edges_nm * 10.0  # MDAnalysis distances are in Angstrom
    shell_vol_nm3 = (4.0 / 3.0) * np.pi * (edges_nm[1:] ** 3 - edges_nm[:-1] ** 3)
    return edges_nm, centers_nm, edges_a, shell_vol_nm3


def compute_density_profile(
    solute: mda.AtomGroup,
    water: mda.AtomGroup,
    start_frame: int,
    end_frame: int,
    edges_a: np.ndarray,
    shell_vol_nm3: np.ndarray,
) -> np.ndarray:
    """Per-atom raw water number-density histogram n(r) (waters/nm^3),
    averaged over all frames in [start_frame, end_frame) of the trajectory
    both AtomGroups belong to.

    Returns an array of shape (len(solute), len(edges_a) - 1).
    """
    hist_sum = np.zeros((len(solute), len(edges_a) - 1))
    n_trajectory = len(solute.universe.trajectory)
    if not 0 <= start_frame < end_frame <= n_trajectory:
        raise ValueError(
            f"Invalid frame range [{start_frame}, {end_frame}) for a trajectory "
            f"of {n_trajectory} frames"
        )
    n_frames = end_frame - start_frame
    for ts in solute.universe.trajectory[start_frame:end_frame]:
        distances = distance_array(
            solute.positions, water.positions, box=ts.dimensions
        )  # Angstrom
        for ai, atom_distances in enumerate(distances):
            counts, _ = np.histogram(atom_distances, bins=edges_a)
            hist_sum[ai] += counts
    mean_hist = hist_sum / n_frames  # mean neighbour count per bin per frame
    return mean_hist / shell_vol_nm3  # number density, waters/nm^3


def fp_from_density_profile(
    n_r: np.ndarray,
    r_centers_nm: np.ndarray,
    norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT,
) -> tuple[float, np.ndarray, float]:
    """Given one atom's raw number-density profile n_r (waters/nm^3 per
    bin) and its radial bin centers, return (FP, g(r), norm).

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
    return fp, g, norm


def fingerprints_from_rdf_table(rdf_df, norm_tail_bins=NORM_TAIL_BINS_DEFAULT):
    """rdf_df: DataFrame with columns atom,r_nm,n_r (the rdf.csv format,
    possibly with extra columns e.g. block - grouped away).
    Returns a DataFrame with columns atom,fp."""
    rows = []
    for atom, sub in rdf_df.groupby("atom", sort=False):
        sub = sub.sort_values("r_nm")
        fp, _, _ = fp_from_density_profile(
            sub["n_r"].to_numpy(), sub["r_nm"].to_numpy(), norm_tail_bins
        )
        rows.append({"atom": atom, "fp": fp})
    return pd.DataFrame(rows)
