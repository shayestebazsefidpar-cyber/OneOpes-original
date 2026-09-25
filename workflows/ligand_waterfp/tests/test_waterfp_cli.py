"""
Unit tests for ligand_waterfp.waterfp.cli's rdf.csv (de)serialization
helpers - the pandas edge around the numpy math in fingerprint.py.
The fingerprint subcommand itself is exercised end-to-end in
test_output_dir_creation.py.
"""

import numpy as np
import pandas as pd

from ligand_waterfp.waterfp.cli import (
    _fingerprints_from_rdf_table,
    _profiles_to_table,
)
from ligand_waterfp.waterfp.fingerprint import NORM_TAIL_BINS_DEFAULT


def test_profiles_to_table_one_row_per_atom_per_bin():
    centers_nm = np.array([0.05, 0.15, 0.25])
    n_r = np.array([[1.0, 2.0, 3.0],
                    [4.0, 5.0, 6.0]])

    df = _profiles_to_table(["C1", "N2"], centers_nm, n_r)

    assert list(df.columns) == ["atom", "r_nm", "n_r"]
    # atom-major ordering: all of C1's bins first, then all of N2's
    assert list(df["atom"]) == ["C1", "C1", "C1", "N2", "N2", "N2"]
    assert np.allclose(df["r_nm"], np.tile(centers_nm, 2))
    assert np.allclose(df["n_r"], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])


def test_fingerprints_from_rdf_table_groups_by_atom():
    # two atoms with flat bulk profiles -> both FPs must be 0
    n_bins = 600
    r_centers = (np.arange(n_bins) + 0.5) * 0.001
    n_r = np.full((2, n_bins), 33.4)
    rdf_df = _profiles_to_table(["A1", "A2"], r_centers, n_r)

    fp_df = _fingerprints_from_rdf_table(rdf_df, NORM_TAIL_BINS_DEFAULT)

    assert list(fp_df.columns) == ["atom", "fp"]
    assert list(fp_df["atom"]) == ["A1", "A2"]
    assert np.allclose(fp_df["fp"], 0.0, atol=1e-8)


def test_table_round_trip_is_lossless():
    rng = np.random.default_rng(3)
    centers_nm = (np.arange(50) + 0.5) * 0.01
    n_r = rng.uniform(20.0, 40.0, size=(3, 50))

    table = _profiles_to_table(["X1", "X2", "X3"], centers_nm, n_r)
    # regroup the long table back into per-atom profiles
    for i, (atom, sub) in enumerate(table.groupby("atom", sort=False)):
        sub = sub.sort_values("r_nm")
        assert np.array_equal(sub["n_r"].to_numpy(), n_r[i])
        assert np.array_equal(sub["r_nm"].to_numpy(), centers_nm)
