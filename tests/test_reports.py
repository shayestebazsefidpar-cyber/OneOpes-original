"""Tests for the `reports` module: the tables and files written from hydration and convergence results."""
import dataclasses
import json

import pandas as pd

from oneopes.convergence import DEFAULT_TOLERANCES, StabilityResult, summarize
from oneopes.reports import (
    fp_ranks,
    transitions_table,
    write_block_tables,
    write_hydration_files,
    write_summary,
)
from oneopes.trajectory_blocks import block_measurement, stack_blocks
from oneopes.waterfp import measure_hydration
from water_systems import random_universe

TOL = DEFAULT_TOLERANCES


def _stabilities(passed):
    """Stability results that only differ in whether they passed."""
    return [StabilityResult(0.0, 0.0, 1.0, p, p, p, p) for p in passed]


def _measured_blocks(n_blocks, labels=None, block_ns=1.0, start_ns=0.0):
    """Hydration of three solute atoms in one-frame blocks of block_ns ns from start_ns."""
    u = random_universe(n_frames=n_blocks)
    solute, water = u.atoms[:3], u.atoms[3:]
    return stack_blocks([block_measurement(measure_hydration(solute, water, range(b, b + 1), labels=labels),
                                           range(b, b + 1), (start_ns + b * block_ns, start_ns + (b + 1) * block_ns))
                         for b in range(n_blocks)])


def test_transitions_table_rows_and_times():
    """Six blocks of 2.5 ns from 12.5 ns: one row per block after the first."""
    stabilities = _stabilities([False, True, True, True, False])
    table = transitions_table(stabilities, _measured_blocks(6, block_ns=2.5, start_ns=12.5))
    assert list(table.columns) == [
        "block", "t_start_ns", "t_end_ns", "t_mid_ns",
        "fp_change", "rdf_nrmsd", "spearman_rho", "fp_pass", "rdf_pass", "spearman_pass", "passed",
        "stable_in_a_row"]
    assert list(table["block"]) == [1, 2, 3, 4, 5]
    assert list(table["t_end_ns"]) == [17.5, 20.0, 22.5, 25.0, 27.5]
    assert list(table["passed"]) == [False, True, True, True, False]
    assert list(table["stable_in_a_row"]) == [0, 1, 2, 3, 0]


def test_a_single_block_has_no_transitions_but_keeps_the_header(tmp_path):
    transitions_table([], _measured_blocks(1)).to_csv(tmp_path / "metrics.csv", index=False)
    assert (tmp_path / "metrics.csv").read_text().startswith("block,t_start_ns,t_end_ns,")
    assert pd.read_csv(tmp_path / "metrics.csv").empty


def test_write_summary_adds_facts_to_the_summary(tmp_path):
    summary = summarize(_stabilities([True] * 3), _measured_blocks(4), 1.0, TOL)
    write_summary(tmp_path / "summary.json", summary, ligand_heavy_atoms=["C1", "O1"])
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["tolerances"] == dataclasses.asdict(TOL)
    assert written["converged_block"] == 3 and written["ligand_heavy_atoms"] == ["C1", "O1"]


def test_fp_rank_one_is_the_most_negative_fp():
    fp = pd.Series({"x": -3.0, "y": -40.0, "z": 2.0})
    assert fp_ranks(fp).to_dict() == {"x": 2, "y": 1, "z": 3}


def test_fp_ranks_per_block_with_ties_in_atom_order():
    fp = pd.DataFrame({0: [-1.0, -1.0, -5.0], 1: [0.0, -2.0, -1.0]}, index=["x", "y", "z"])
    assert fp_ranks(fp).to_dict() == {0: {"x": 2, "y": 3, "z": 1}, 1: {"x": 3, "y": 1, "z": 2}}


def test_write_block_tables(tmp_path):
    """Five CSVs; wide tables keep the atoms' order (not alphabetical) and rank each block's FP."""
    write_block_tables(tmp_path, _measured_blocks(4, labels=["c", "a", "b"]), _stabilities([True] * 3))

    fp_long = pd.read_csv(tmp_path / "fp_values_per_block.csv")
    assert list(fp_long.columns) == ["block", "t_start_ns", "t_end_ns", "t_mid_ns", "atom", "FP"]
    assert len(fp_long) == 4 * 3
    assert len(pd.read_csv(tmp_path / "convergence_metrics.csv")) == 3
    rdf = pd.read_csv(tmp_path / "rdf_profiles_per_block.csv")
    assert list(rdf.columns) == ["block", "atom", "r_nm", "g_r"]
    assert rdf["block"].nunique() == 4
    fp = pd.read_csv(tmp_path / "fp_values_by_block_wide.csv", index_col="atom")
    ranks = pd.read_csv(tmp_path / "fp_ranking_by_block_wide.csv", index_col="atom")
    assert list(fp.index) == ["c", "a", "b"]
    assert list(fp.columns) == ["0", "1", "2", "3"]
    assert ranks.equals(fp.rank().astype(int))


def test_write_hydration_files(tmp_path):
    hydration = _measured_blocks(1, labels=["c", "a", "b"]).isel(block=0)
    write_hydration_files(hydration, tmp_path / "selection")

    rdf = pd.read_csv(tmp_path / "selection" / "rdf.csv")
    assert list(rdf.columns) == ["atom", "r_nm", "n_r"]
    assert len(rdf) == 3 * hydration.sizes["r_nm"]
    fingerprints = pd.read_csv(tmp_path / "selection" / "fingerprints.csv")
    assert list(fingerprints.columns) == ["atom", "fp"]
    assert list(fingerprints["atom"]) == ["c", "a", "b"]
