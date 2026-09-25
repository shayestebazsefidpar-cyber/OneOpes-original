"""
Unit tests for ligand_waterfp.waterfp.fingerprint's pure math - no
trajectory needed. compute_density_profile() itself is exercised
indirectly by the real-data validation described in tests/README.md,
since it needs an actual MDAnalysis Universe.
"""

import MDAnalysis as mda
import numpy as np
import pytest
from ligand_waterfp.waterfp.fingerprint import (
    compute_density_profile,
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


def _tiny_universe():
    """3 frames of 1 'solute' atom at the origin and 2 'waters' held at
    fixed distances (5.5 A and 12.5 A, mid-bin for 1 A bins) in a 100 A
    cubic box, all in memory."""
    n_frames = 3
    coords = np.zeros((n_frames, 3, 3), dtype=np.float32)
    coords[:, 1] = [5.5, 0.0, 0.0]
    coords[:, 2] = [0.0, 12.5, 0.0]
    u = mda.Universe.empty(3, trajectory=True)
    u.load_new(coords, dimensions=np.tile([100, 100, 100, 90, 90, 90], (n_frames, 1)))
    return u


def test_compute_density_profile_counts_waters_in_the_right_shells():
    u = _tiny_universe()
    solute, water = u.atoms[:1], u.atoms[1:]
    # 20 bins of 0.1 nm -> one bin edge every 1 A
    _, _, edges_a, shell_vol_nm3 = make_bins(rmax_nm=2.0, binwidth_nm=0.1)

    n_r = compute_density_profile(solute, water, 0, 3, edges_a, shell_vol_nm3)

    assert n_r.shape == (1, 20)
    counts = n_r[0] * shell_vol_nm3  # back to mean neighbour count per frame
    assert counts[5] == pytest.approx(1.0)  # the water 5.5 A away -> bin [5, 6) A
    assert counts[12] == pytest.approx(1.0)  # the water 12.5 A away -> bin [12, 13) A
    assert np.count_nonzero(counts) == 2


def test_compute_density_profile_rejects_bad_frame_ranges():
    u = _tiny_universe()
    solute, water = u.atoms[:1], u.atoms[1:]
    _, _, edges_a, shell_vol_nm3 = make_bins(rmax_nm=2.0, binwidth_nm=0.1)

    with pytest.raises(ValueError, match="frame range"):
        compute_density_profile(solute, water, 2, 2, edges_a, shell_vol_nm3)  # empty
    with pytest.raises(ValueError, match="frame range"):
        compute_density_profile(
            solute, water, 0, 4, edges_a, shell_vol_nm3
        )  # past the end


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
    fp, g, norm = fp_from_density_profile(n_r, r_centers)
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
    fp, g, norm = fp_from_density_profile(n_r, r_centers)
    assert g[:50].tolist() == [0.0] * 50
    assert fp < 0


def test_fp_rejects_profile_that_never_reaches_bulk():
    # all-zero tail -> norm = 0 -> must raise instead of returning NaN FPs
    n_r, r_centers = _uniform_bulk_profile()
    n_r = n_r.copy()
    n_r[-500:] = 0.0
    with pytest.raises(ValueError, match="[Bb]ulk"):
        fp_from_density_profile(n_r, r_centers)
