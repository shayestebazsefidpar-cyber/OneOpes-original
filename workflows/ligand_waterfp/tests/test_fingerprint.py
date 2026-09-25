"""
Unit tests for ligand_waterfp.waterfp.fingerprint's pure math - no
trajectory needed. compute_density_profile() itself is exercised
indirectly by the real-data validation described in tests/README.md,
since it needs an actual MDAnalysis Universe.
"""

import numpy as np
import pandas as pd
import pytest

from ligand_waterfp.waterfp.fingerprint import (
    fingerprints_from_rdf_table,
    fp_from_density_profile,
    make_bins,
)


def test_make_bins_matches_waterfp_default_settings():
    edges_nm, centers_nm, edges_a, shell_vol_nm3 = make_bins(
        rmax_nm=2.001, binwidth_nm=0.001
    )

    assert len(edges_nm) == 2002  # 2001 bins -> 2002 edges
    assert len(centers_nm) == 2001
    assert len(shell_vol_nm3) == 2001
    assert np.isclose(edges_nm[0], 0.0)
    assert np.isclose(edges_nm[-1], 2.001)
    # MDAnalysis distances are in Angstrom - edges_a must be a 10x scale of edges_nm
    assert np.allclose(edges_a, edges_nm * 10.0)


def test_make_bins_shell_volumes_are_positive_and_increasing():
    _, _, _, shell_vol_nm3 = make_bins(rmax_nm=1.0, binwidth_nm=0.1)
    assert np.all(shell_vol_nm3 > 0)
    # spherical shell volume grows with radius for equal-width bins
    assert np.all(np.diff(shell_vol_nm3) > 0)


def test_make_bins_custom_binwidth():
    edges_nm, centers_nm, _, _ = make_bins(rmax_nm=1.0, binwidth_nm=0.1)
    assert len(centers_nm) == 10
    assert np.allclose(np.diff(edges_nm), 0.1)


def _uniform_bulk_profile(n_bins=2001, binwidth_nm=0.001, value=33.4):
    """A flat n(r) profile at the bulk density everywhere - g(r) == 1
    everywhere, so the WaterFP integrand (g*ln(g) - g + 1) is exactly 0 at
    every bin: this should integrate to FP == 0 (no hydration perturbation
    relative to bulk, by construction)."""
    r_centers = (np.arange(n_bins) + 0.5) * binwidth_nm
    n_r = np.full(n_bins, value)
    return n_r, r_centers


def test_fp_is_zero_for_uniform_bulk_density():
    n_r, r_centers = _uniform_bulk_profile()
    fp, g, norm = fp_from_density_profile(n_r, r_centers, binwidth_nm=0.001)
    assert norm == pytest.approx(33.4)
    assert np.allclose(g, 1.0)
    assert abs(fp) < 1e-8


def test_fp_is_nonzero_for_a_depleted_shell():
    # bulk everywhere except a fully-depleted (g=0) first shell -> the
    # g=0 limiting case (integrand = 1, i.e. -2*pi*norm*r^2) must
    # contribute a nonzero, negative FP.
    n_r, r_centers = _uniform_bulk_profile()
    n_r = n_r.copy()
    n_r[:50] = 0.0
    fp, g, norm = fp_from_density_profile(n_r, r_centers, binwidth_nm=0.001)
    assert g[:50].tolist() == [0.0] * 50
    assert fp < 0


def test_fingerprints_from_rdf_table_groups_by_atom():
    n_r, r_centers = _uniform_bulk_profile(n_bins=600)
    rows = []
    for atom in ["A1", "A2"]:
        for r, n in zip(r_centers, n_r):
            rows.append({"atom": atom, "r_nm": r, "n_r": n})
    rdf_df = pd.DataFrame(rows)

    fp_df = fingerprints_from_rdf_table(rdf_df, binwidth_nm=0.001)

    assert set(fp_df["atom"]) == {"A1", "A2"}
    assert len(fp_df) == 2
    assert np.allclose(fp_df["fp"], 0.0, atol=1e-8)
