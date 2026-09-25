"""
Water-oxygen radial density profile n(r) for a set of solute (ligand) heavy
atoms, computed directly from an MDAnalysis-readable trajectory.

Matches the reference WaterFP repository's own RDF settings
(github.com/valeriorizzi/WaterFP, Scripts/calc_rdf.sh: `gmx rdf -bin 0.001
-norm number_density -rmax 2.001`), but computed here per-frame-range with
MDAnalysis instead of `gmx rdf` on a fixed trajectory - so the same code
can be called once (the `ligand-waterfp run`/`rdf` subcommands in cli.py,
over a whole converged trajectory) or repeatedly per block
(`ligand_waterfp.convergence.monitor_convergence`, for online convergence
checking).

This module has no notion of "convergence," "blocks," or "stability" - it
only computes n(r). See calculate_fingerprint.py for turning n(r) into a
WaterFP value, and the `convergence` subpackage for the block-wise
convergence loop built on top of both.
"""

import MDAnalysis as mda
import numpy as np
from MDAnalysis.lib.distances import distance_array

RDF_RMAX_NM_DEFAULT = 2.001
RDF_BINWIDTH_NM_DEFAULT = 0.001


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
