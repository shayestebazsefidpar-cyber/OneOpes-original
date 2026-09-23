"""
Unit tests for ligand_waterfp.waterfp.calculate_rdf's pure-math helper
(make_bins) - no trajectory/MDAnalysis needed. compute_density_profile()
itself is exercised indirectly by the real-data validation described in
tests/README.md, since it needs an actual MDAnalysis Universe.
"""
import numpy as np

from ligand_waterfp.waterfp.calculate_rdf import make_bins


def test_make_bins_matches_waterfp_default_settings():
    edges_nm, centers_nm, edges_a, shell_vol_nm3 = make_bins(rmax_nm=2.001, binwidth_nm=0.001)

    assert len(edges_nm) == 2002          # 2001 bins -> 2002 edges
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
