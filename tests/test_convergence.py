"""Tests for the shared `convergence` module: one transition's stability, and convergence over a run."""
import dataclasses

import numpy as np
import pandas as pd
import pytest
from MDAnalysis.analysis.results import Results

from convergence import (
    DEFAULT_TOLERANCES,
    StabilityResult,
    Tolerances,
    check_stability,
    compare_blocks,
    first_convergence,
    fp_ranks,
    nrmsd,
    stable_in_a_row,
    transitions_table,
)

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
    stabilities = _stabilities([False, True, True, True, False])  # converged at transition 3 (block 3 -> 4)
    block_times = [(12.5 + 2.5 * b, 12.5 + 2.5 * (b + 1)) for b in range(6)]
    table = transitions_table(stabilities, block_times, n_stable=3)
    assert list(table.columns) == [
        "transition", "previous_block", "block", "block_start_ns", "block_end_ns",
        "FP_relative_change", "FP_pass", "RDF_nRMSD", "RDF_pass", "Spearman_rho",
        "Spearman_pass", "all_criteria_pass", "stable_transition_count", "convergence_reached"]
    assert list(table["transition"]) == ["0->1", "1->2", "2->3", "3->4", "4->5"]
    assert list(table["block_end_ns"]) == [17.5, 20.0, 22.5, 25.0, 27.5]
    assert list(table["stable_transition_count"]) == [0, 1, 2, 3, 0]
    assert list(table["convergence_reached"]) == [False, False, False, True, True]


def test_fp_rank_one_is_the_most_negative_fp():
    fp = pd.Series({"x": -3.0, "y": -40.0, "z": 2.0})
    assert fp_ranks(fp).to_dict() == {"x": 2, "y": 1, "z": 3}


def test_fp_ranks_per_block_with_ties_in_atom_order():
    fp = pd.DataFrame({0: [-1.0, -1.0, -5.0], 1: [0.0, -2.0, -1.0]}, index=["x", "y", "z"])
    assert fp_ranks(fp).to_dict() == {0: {"x": 2, "y": 3, "z": 1}, 1: {"x": 3, "y": 1, "z": 2}}
