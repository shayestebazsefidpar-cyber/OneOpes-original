"""Tests for the shared `convergence` module: one transition's stability, and convergence over a run."""
import dataclasses
import json

import numpy as np
import pandas as pd
import pytest
from MDAnalysis.analysis.results import Results

from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    ConvergenceSummary,
    StabilityResult,
    Tolerances,
    check_stability,
    compare_blocks,
    first_convergence,
    fp_ranks,
    nrmsd,
    stable_in_a_row,
    summarize,
    transitions_table,
    write_block_tables,
    write_summary,
)
from water_systems import random_universe
from oneopes.waterfp import WaterFP

TOL = DEFAULT_TOLERANCES


def _same_g(n_atoms):
    """Identical g(r) for every atom (and, reused across blocks, nRMSD = 0)."""
    return np.tile([0.0, 0.5, 1.0, 1.0, 1.0], (n_atoms, 1))


def _block(fp, g):
    """A block's WaterFP results."""
    return Results(fp=np.asarray(fp, dtype=float), g=np.asarray(g, dtype=float))


def _fp_run(*fps):
    """WaterFP results of consecutive blocks with the given FP values and identical g(r)."""
    return [_block(fp, _same_g(len(fp))) for fp in fps]


def _stabilities(passed):
    """Stability results that only differ in whether they passed."""
    return [StabilityResult(0.0, 0.0, 1.0, p, p, p, p) for p in passed]


# --- one transition ---------------------------------------------------------

def test_default_tolerances():
    assert (TOL.fp_rel_tol, TOL.rdf_nrmsd_tol, TOL.spearman_tol, TOL.n_stable) == (0.10, 0.15, 0.90, 3)


def test_stable_transition_passes_all_three_criteria():
    g = _same_g(3)
    result = check_stability(_block([-10.5, -8.3, -5.1], g), _block([-10.0, -8.0, -5.0], g), TOL)
    assert (result.fp_pass, result.rdf_pass, result.spearman_pass, result.passed) == (True, True, True, True)
    assert result.fp_change == pytest.approx(0.05)  # 5%, 3.75%, 2%: the largest counts
    assert result.rdf_nrmsd == pytest.approx(0.0)
    assert result.spearman_rho == pytest.approx(1.0)


def test_fp_change_above_threshold_alone_makes_transition_unstable():
    g = _same_g(3)
    result = check_stability(_block([-11.5, -8.1, -5.05], g), _block([-10.0, -8.0, -5.0], g), TOL)
    assert (result.fp_pass, result.rdf_pass, result.spearman_pass, result.passed) == (False, True, True, False)
    assert result.fp_change == pytest.approx(0.15)


def test_rdf_nrmsd_above_threshold_alone_makes_transition_unstable():
    prev_g = _same_g(3)
    g = prev_g.copy()
    g[0] = [1.0, 1.0, 1.0, 0.5, 0.0]  # one atom's g(r) changes shape
    result = check_stability(_block([-10.2, -8.1, -5.05], g), _block([-10.0, -8.0, -5.0], prev_g), TOL)
    assert (result.fp_pass, result.rdf_pass, result.spearman_pass, result.passed) == (True, False, True, False)
    assert result.rdf_nrmsd > TOL.rdf_nrmsd_tol


def test_reversed_ranking_alone_makes_transition_unstable():
    # closely spaced FP values: every atom changes < 10%, yet the order fully reverses
    g = _same_g(4)
    result = check_stability(_block([-10.25, -10.15, -10.05, -9.95], g), _block([-10.0, -10.1, -10.2, -10.3], g), TOL)
    assert (result.fp_pass, result.rdf_pass, result.spearman_pass, result.passed) == (True, True, False, False)
    assert result.spearman_rho == pytest.approx(-1.0)


def test_nrmsd_gives_one_value_per_atom():
    prev_g = _same_g(2)
    g = prev_g.copy()
    g[1] = [1.0, 1.0, 1.0, 0.5, 0.0]
    per_atom = nrmsd(g, prev_g)
    assert per_atom.shape == (2,)
    assert per_atom[0] == 0.0
    assert per_atom[1] == pytest.approx(nrmsd(g[1], prev_g[1]))  # same as one atom's 1-D curves


def test_results_and_tolerances_are_frozen():
    block = _block([-10.0, -5.0], _same_g(2))
    with pytest.raises(dataclasses.FrozenInstanceError):
        check_stability(block, block, TOL).passed = False
    with pytest.raises(dataclasses.FrozenInstanceError):
        Tolerances(0.1, 0.15, 0.9, 3).n_stable = 5


# --- a run of blocks --------------------------------------------------------

def test_compare_blocks_scores_each_block_against_the_previous_one():
    fp = np.array([-30.0, -22.0, -15.0, -9.0])
    stabilities = compare_blocks(_fp_run(fp, fp, fp * 1.15), TOL)
    assert [s.passed for s in stabilities] == [True, False]  # 0->1 identical, 1->2 FP +15 %


def test_steady_fp_growth_fails_only_the_fp_criterion():
    fp = np.array([-30.0, -22.0, -15.0, -9.0])
    stabilities = compare_blocks(_fp_run(*(fp * 1.15 ** b for b in range(6))), TOL)
    assert all(s.spearman_pass and s.rdf_pass and not s.fp_pass for s in stabilities)
    assert first_convergence(stabilities, TOL.n_stable) is None


def test_stable_in_a_row_resets_on_an_unstable_transition():
    passed = [False] * 4 + [True] * 3 + [False] + [True] * 2
    assert stable_in_a_row(_stabilities(passed)) == [0, 0, 0, 0, 1, 2, 3, 0, 1, 2]


def test_first_convergence_is_the_transition_completing_the_first_streak():
    stabilities = _stabilities([True, True, False, True, True, True, False, True, True, True])
    assert first_convergence(stabilities, 3) == 5
    assert first_convergence(stabilities[:5], 3) is None


def test_transitions_table_rows_and_times():
    """Six blocks of 2.5 ns from 12.5 ns: one row per block after the first."""
    stabilities = _stabilities([False, True, True, True, False])
    block_times = [(12.5 + 2.5 * b, 12.5 + 2.5 * (b + 1)) for b in range(6)]
    table = transitions_table(stabilities, block_times)
    assert list(table.columns) == [
        "block", "t_start_ns", "t_end_ns", "t_mid_ns",
        "fp_change", "rdf_nrmsd", "spearman_rho", "fp_pass", "rdf_pass", "spearman_pass", "passed",
        "stable_in_a_row"]
    assert list(table["block"]) == [1, 2, 3, 4, 5]
    assert list(table["t_end_ns"]) == [17.5, 20.0, 22.5, 25.0, 27.5]
    assert list(table["passed"]) == [False, True, True, True, False]
    assert list(table["stable_in_a_row"]) == [0, 1, 2, 3, 0]


def test_a_single_block_has_no_transitions_but_keeps_the_header(tmp_path):
    transitions_table([], [(0.0, 5.0)]).to_csv(tmp_path / "metrics.csv", index=False)
    assert (tmp_path / "metrics.csv").read_text().startswith("block,t_start_ns,t_end_ns,")
    assert pd.read_csv(tmp_path / "metrics.csv").empty


def test_summarize_converged_run():
    """Converged at transition 2->3 (block 3); of the two later transitions one failed."""
    stabilities = _stabilities([True, True, True, False, True])
    block_times = [(2.0 * b, 2.0 * (b + 1)) for b in range(6)]
    summary = summarize(stabilities, block_times, 2.0, TOL)
    assert summary == ConvergenceSummary(block_ns=2.0, tolerances=TOL, n_blocks=6, converged=True,
                                         converged_block=3, converged_ns=8.0, unstable_after_convergence=1,
                                         stable_in_a_row_at_end=1)


def test_summarize_unconverged_and_single_block_runs():
    summary = summarize(_stabilities([True, False]), [(0.0, 1.0)] * 3, 1.0, TOL)
    assert (summary.converged, summary.converged_block, summary.converged_ns) == (False, None, None)
    assert summarize([], [(0.0, 1.0)], 1.0, TOL).stable_in_a_row_at_end == 0


def test_write_summary_adds_facts_to_the_summary(tmp_path):
    summary = summarize(_stabilities([True] * 3), [(0.0, 1.0)] * 4, 1.0, TOL)
    write_summary(tmp_path / "summary.json", summary, ligand_heavy_atoms=["C1", "O1"])
    written = json.loads((tmp_path / "summary.json").read_text())
    assert written["tolerances"] == dataclasses.asdict(TOL)
    assert written["converged_block"] == 3 and written["ligand_heavy_atoms"] == ["C1", "O1"]


def test_fp_rank_one_is_the_most_negative_fp():
    fp = pd.Series({"x": -3.0, "y": -40.0, "z": 2.0})
    assert fp_ranks(fp).to_dict() == {"x": 2, "y": 1, "z": 3}


def test_write_block_tables(tmp_path):
    """Five CSVs; wide tables keep the atoms' order (not alphabetical) and rank each block's FP."""
    u = random_universe(n_frames=4)
    blocks = [WaterFP(u.atoms[:3], u.atoms[3:]).run(start=i, stop=i + 1) for i in range(4)]
    write_block_tables(tmp_path, blocks, [(i, i + 1.0) for i in range(4)], TOL, labels=["c", "a", "b"])

    assert len(pd.read_csv(tmp_path / "fp_values_per_block.csv")) == 4 * 3
    assert len(pd.read_csv(tmp_path / "convergence_metrics.csv")) == 3
    assert pd.read_csv(tmp_path / "rdf_profiles_per_block.csv")["block"].nunique() == 4
    fp = pd.read_csv(tmp_path / "fp_values_by_block_wide.csv", index_col="atom")
    ranks = pd.read_csv(tmp_path / "fp_ranking_by_block_wide.csv", index_col="atom")
    assert list(fp.index) == ["c", "a", "b"]
    assert ranks.equals(fp.rank().astype(int))


def test_fp_ranks_per_block_with_ties_in_atom_order():
    fp = pd.DataFrame({0: [-1.0, -1.0, -5.0], 1: [0.0, -2.0, -1.0]}, index=["x", "y", "z"])
    assert fp_ranks(fp).to_dict() == {0: {"x": 2, "y": 3, "z": 1}, 1: {"x": 3, "y": 1, "z": 2}}
