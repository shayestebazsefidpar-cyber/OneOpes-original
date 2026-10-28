"""Convergence: the ligand-side all-three rule, 3 consecutive transitions, reported retrospectively."""

import numpy as np
from MDAnalysis.analysis.results import Results

from convergence import (
    DEFAULT_TOLERANCES,
    StabilityResult,
    compare_blocks,
    first_convergence,
    stable_in_a_row,
    transitions_table,
)
from protein_waterfp.convergence import fp_ranks

N_ATOMS = 6
G = np.tile(np.linspace(0.0, 2.0, 50), (N_ATOMS, 1))  # same g(r) for every atom


def _fp(scale=1.0, swap=False):
    base = np.array([-30.0, -22.0, -15.0, -9.0, -4.0, -1.0])
    if swap:
        base = base[::-1]
    return base * scale


def _blocks(fps, g=G):
    """WaterFP results of consecutive blocks with the given FP values."""
    return [Results(fp=fp, g=g) for fp in fps]


def _stabilities(passed):
    """Stability results that only differ in whether they passed."""
    return [StabilityResult(0.0, 0.0, 1.0, p, p, p, p) for p in passed]


def test_reused_ligand_side_thresholds():
    t = DEFAULT_TOLERANCES
    assert (t.fp_rel_tol, t.rdf_nrmsd_tol, t.spearman_tol, t.n_stable) == (0.10, 0.15, 0.90, 3)


def test_compare_blocks_scores_each_block_against_the_previous_one():
    stabilities = compare_blocks(_blocks([_fp(), _fp(), _fp(1.15)]), DEFAULT_TOLERANCES)
    assert [s.passed for s in stabilities] == [True, False]  # 0->1 identical, 1->2 FP +15 %


def test_all_three_criteria_required_fp_change_alone_breaks_stability():
    stabilities = compare_blocks(_blocks([_fp(1.15 ** b) for b in range(6)]), DEFAULT_TOLERANCES)
    assert all(s.spearman_pass and s.rdf_pass and not s.fp_pass for s in stabilities)


def test_rdf_change_alone_breaks_stability():
    blocks = [Results(fp=_fp(), g=G * (1.0 + 0.5 * (b % 2))) for b in range(6)]
    assert any(not s.rdf_pass for s in compare_blocks(blocks, DEFAULT_TOLERANCES))


def test_spearman_only_stability_is_insufficient():
    stabilities = compare_blocks(_blocks([_fp(1.0 + 0.3 * b) for b in range(8)]), DEFAULT_TOLERANCES)
    assert all(s.spearman_rho >= 0.9 for s in stabilities)
    assert first_convergence(stabilities, 3) is None


def test_stable_in_a_row_resets_on_an_unstable_transition():
    passed = [False] * 4 + [True] * 3 + [False] + [True] * 2
    assert stable_in_a_row(_stabilities(passed)) == [0, 0, 0, 0, 1, 2, 3, 0, 1, 2]


def test_first_convergence_is_the_transition_completing_the_first_streak():
    stabilities = _stabilities([True, True, False, True, True, True, False, True, True, True])
    assert first_convergence(stabilities, 3) == 5
    assert first_convergence(stabilities[:5], 3) is None


def test_rank_one_is_most_negative_fp():
    ranks = fp_ranks({"x": -3.0, "y": -40.0, "z": 2.0}, ["x", "y", "z"])
    assert ranks.to_dict() == {"x": 2.0, "y": 1.0, "z": 3.0}


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
