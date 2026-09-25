"""
WaterFP hydration fingerprint (FP), computed from a raw water number-density
profile n(r) (e.g. as produced by calculate_rdf.py).

Reimplements the reference WaterFP repository's fp.py exactly
(github.com/valeriorizzi/WaterFP, Scripts/fp.py):

    norm  = mean of the last 500 of 2001 radial bins (empirical bulk-water
            density plateau; no external/theoretical bulk density assumed)
    g(r)  = n(r) / norm                          (dimensionless RDF)
    FP    = integral_0^{rmax}  -2*pi*norm*(g*ln(g) - g + 1)*r^2 dr
            (trapezoidal rule; where g=0 the integrand's well-defined
            limit, 1, is substituted for (g*ln(g)-g+1))

This integrand is the standard water excess/relative-entropy density
relative to bulk (the functional form used in inhomogeneous solvation
theory / GIST-style analyses): it grows wherever a solute atom's presence
makes local water systematically denser or sparser than bulk. It is not a
coordination number - two atoms with similar coordination can have
different FP if the shape of their water shell differs from bulk.

This module has no notion of "convergence" or "ranking stability" - it
only turns n(r) into a scalar FP per atom. See
the `convergence` subpackage for the block-wise convergence loop
built on top of this and calculate_rdf.py.
"""

import numpy as np
import pandas as pd

_trapz = getattr(np, "trapezoid", None) or np.trapz

NORM_TAIL_BINS_DEFAULT = 500


def fp_from_density_profile(
    n_r, r_centers_nm, binwidth_nm, norm_tail_bins=NORM_TAIL_BINS_DEFAULT
):
    """Given one atom's raw number-density profile n_r (waters/nm^3 per
    bin) and its radial bin centers, return (FP, g(r), norm)."""
    norm = float(np.mean(n_r[-norm_tail_bins:]))
    g = n_r / norm
    r = r_centers_nm
    integrand = np.empty_like(g)
    mask0 = g == 0
    integrand[mask0] = (
        -2 * np.pi * norm * (r[mask0] ** 2)
    )  # lim g->0 of (g ln g - g + 1) = 1
    gnz = g[~mask0]
    integrand[~mask0] = (
        -2 * np.pi * norm * (gnz * np.log(gnz) - gnz + 1) * (r[~mask0] ** 2)
    )
    fp = float(_trapz(integrand, dx=binwidth_nm))
    return fp, g, norm


def fingerprints_from_rdf_table(
    rdf_df, binwidth_nm, norm_tail_bins=NORM_TAIL_BINS_DEFAULT
):
    """rdf_df: DataFrame with columns atom,r_nm,n_r (calculate_rdf.py's
    output format, possibly with extra columns e.g. block - grouped away).
    Returns a DataFrame with columns atom,FP."""
    rows = []
    for atom, sub in rdf_df.groupby("atom", sort=False):
        sub = sub.sort_values("r_nm")
        fp, _, _ = fp_from_density_profile(
            sub["n_r"].to_numpy(), sub["r_nm"].to_numpy(), binwidth_nm, norm_tail_bins
        )
        rows.append({"atom": atom, "fp": fp})
    return pd.DataFrame(rows)
