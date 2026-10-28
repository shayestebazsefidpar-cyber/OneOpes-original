"""
Unit tests for ligand_waterfp.waterfp.calculate_fingerprint - pure math,
no trajectory/MDAnalysis needed.
"""
import numpy as np
import pandas as pd
import pytest

from ligand_waterfp.waterfp.calculate_fingerprint import (
    fp_from_density_profile,
    fingerprints_from_rdf_table,
)


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
