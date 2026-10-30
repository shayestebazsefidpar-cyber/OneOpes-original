"""Tests for the `reports` module: the tables and files written from hydration and convergence results."""
import dataclasses
import json

import pandas as pd

from oneopes.block_run import BlockOptions, block_run
from oneopes.convergence import (
    CHECK_VARIABLES,
    UNCHECKED,
    ConvergenceCriteria,
    FpPrecision,
    check_every_block,
    production_estimate,
    with_checks,
)
from oneopes.reports import (
    fp_ranks,
    write_block_tables,
    write_hydration_files,
    write_summary,
)
from oneopes.trajectory_blocks import stack_blocks
from oneopes.waterfp import measure_hydration
from synthetic_blocks import COARSE_OPTIONS, one_frame_blocks, water_blocks
from water_systems import random_universe


def _measured_blocks(n_blocks, labels=None, block_ns=1.0, start_ns=0.0):
    """Hydration of three solute atoms in one-frame blocks of block_ns ns from start_ns, not checked."""
    u = random_universe(n_frames=n_blocks)
    solute, water = u.atoms[:3], u.atoms[3:]
    blocks = stack_blocks([block.label(measure_hydration(solute, water, block.frames, labels=labels))
                           for block in one_frame_blocks(n_blocks, block_ns, start_ns)])
    return with_checks(blocks, [UNCHECKED] * n_blocks)


def test_convergence_metrics_has_the_check_after_each_block(tmp_path):
    """Six blocks of 2.5 ns from 12.5 ns: one row per block."""
    write_block_tables(tmp_path, _measured_blocks(6, block_ns=2.5, start_ns=12.5))
    table = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert list(table.columns) == ["block", *CHECK_VARIABLES, "passes_in_a_row", "t_start_ns", "t_end_ns", "n_frames",
                                   "t_mid_ns"]
    assert list(table["block"]) == [0, 1, 2, 3, 4, 5]
    assert list(table["t_end_ns"]) == [15.0, 17.5, 20.0, 22.5, 25.0, 27.5]


def test_write_summary_adds_the_run_facts_to_the_summary(tmp_path):
    options = BlockOptions(criteria=ConvergenceCriteria(min_blocks=4))
    goal = FpPrecision(rel_se_tol=1.0)
    blocks = water_blocks([0.5] * 6)
    run = block_run(blocks, check_every_block(blocks, options.criteria, COARSE_OPTIONS, goal), options,
                    COARSE_OPTIONS, goal, ligand_heavy_atoms=["C1", "O1"])
    write_summary(tmp_path / "summary.json", run)
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["criteria"] == dataclasses.asdict(options.criteria)
    assert written["converged_block"] == 4 and written["ligand_heavy_atoms"] == ["C1", "O1"]
    assert written["goal"] == goal.description
    assert "block_ns" in written and "n_passes" not in written


def test_fp_rank_one_is_the_most_negative_fp():
    fp = pd.Series({"x": -3.0, "y": -40.0, "z": 2.0})
    assert fp_ranks(fp).to_dict() == {"x": 2, "y": 1, "z": 3}


def test_fp_ranks_per_block_with_ties_in_atom_order():
    fp = pd.DataFrame({0: [-1.0, -1.0, -5.0], 1: [0.0, -2.0, -1.0]}, index=["x", "y", "z"])
    assert fp_ranks(fp).to_dict() == {0: {"x": 2, "y": 3, "z": 1}, 1: {"x": 3, "y": 1, "z": 2}}


def test_write_block_tables(tmp_path):
    """Five CSVs; wide tables keep the atoms' order (not alphabetical) and rank each block's FP."""
    write_block_tables(tmp_path, _measured_blocks(4, labels=["c", "a", "b"]))

    fp_long = pd.read_csv(tmp_path / "fp_values_per_block.csv")
    assert list(fp_long.columns) == ["block", "atom", "atom_index", "t_start_ns", "t_end_ns", "n_frames", "t_mid_ns",
                                     "FP"]
    assert len(fp_long) == 4 * 3
    assert len(pd.read_csv(tmp_path / "convergence_metrics.csv")) == 4
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


def test_the_fingerprints_of_a_production_estimate_have_its_errors(tmp_path):
    write_hydration_files(production_estimate(water_blocks([0.5] * 4), COARSE_OPTIONS), tmp_path)
    fingerprints = pd.read_csv(tmp_path / "fingerprints.csv")
    assert list(fingerprints.columns) == ["atom", "fp", "fp_se", "fp_uncorrected", "drift_z"]
