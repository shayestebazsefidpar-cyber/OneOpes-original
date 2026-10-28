"""Convergence: the ligand-side all-three rule, 3 consecutive transitions, reported retrospectively."""

import numpy as np
from MDAnalysis.analysis.results import Results

from convergence import DEFAULT_TOLERANCES, Block, ConvergenceTracker, transitions_table
from protein_waterfp.convergence import fp_ranks

N_ATOMS = 6
G = np.tile(np.linspace(0.0, 2.0, 50), (N_ATOMS, 1))  # same g(r) for every atom


def _fp(scale=1.0, swap=False):
    base = np.array([-30.0, -22.0, -15.0, -9.0, -4.0, -1.0])
    if swap:
        base = base[::-1]
    return base * scale


def _block(index, fp, g=G, t0=0.0, block_ns=5.0):
    """Block ``index`` of a trajectory starting at t0, with the given WaterFP results."""
    return Block(index, t0 + block_ns * index, t0 + block_ns * (index + 1), Results(fp=fp, g=g))


def _run(fps, t0=0.0, block_ns=5.0):
    tr = ConvergenceTracker()
    for b, fp in enumerate(fps):
        tr.add_block(_block(b, fp, t0=t0, block_ns=block_ns))
    return tr


def test_reused_ligand_side_thresholds():
    t = DEFAULT_TOLERANCES
    assert (t.fp_rel_tol, t.rdf_nrmsd_tol, t.spearman_tol, t.n_stable) == (0.10, 0.15, 0.90, 3)


def test_three_consecutive_stable_transitions_required():
    tr = _run([_fp()] * 3)  # 2 stable transitions only
    assert tr.first_convergence is None
    tr.add_block(_block(3, _fp()))  # third stable transition
    assert tr.first_convergence.block.index == 3


def test_all_three_criteria_required_fp_change_alone_breaks_stability():
    tr = _run([_fp(1.15 ** b) for b in range(6)])  # same ranking, same g(r), FP changes 15% per block
    assert all(t.stability.spearman_pass and t.stability.rdf_pass and not t.stability.fp_pass
               for t in tr.transitions)
    assert tr.first_convergence is None


def test_rdf_change_alone_breaks_stability():
    tr = ConvergenceTracker()
    for b in range(6):
        tr.add_block(_block(b, _fp(), G * (1.0 + 0.5 * (b % 2))))
    assert any(not t.stability.rdf_pass for t in tr.transitions) and tr.first_convergence is None


def test_spearman_only_stability_is_insufficient():
    tr = _run([_fp(1.0 + 0.3 * b) for b in range(8)])
    assert all(t.stability.spearman_rho >= 0.9 for t in tr.transitions)
    assert tr.first_convergence is None


def test_rank_one_is_most_negative_fp():
    ranks = fp_ranks({"x": -3.0, "y": -40.0, "z": 2.0}, ["x", "y", "z"])
    assert ranks.to_dict() == {"x": 2.0, "y": 1.0, "z": 3.0}


def test_first_sustained_convergence_is_end_of_block_completing_the_streak():
    # transitions 0->1..3->4 unstable (FP +15 %/block), then stable from block 5 onward
    fps = [_fp(1.15 ** b) for b in range(5)] + [_fp(1.15 ** 4)] * 6
    tr = _run(fps)
    assert [t.stability.passed for t in tr.transitions] == [False] * 4 + [True] * 6
    assert tr.first_convergence.block.index == 7 and tr.first_convergence.block.end_ns == 40.0
    assert [t.stable_in_a_row for t in tr.transitions] == [0, 0, 0, 0, 1, 2, 3, 4, 5, 6]
    assert list(transitions_table(tr)["convergence_reached"]) == [False] * 6 + [True] * 4


def test_every_block_is_analysed_after_convergence_and_later_breaks_are_reported():
    fps = [_fp()] * 5 + [_fp(swap=True)] + [_fp(swap=True)] * 3  # converged at block 3, broken at 4->5
    tr = _run(fps, t0=12.5, block_ns=2.5)
    assert len(tr.transitions) == len(fps) - 1  # nothing truncated at convergence
    assert tr.first_convergence.block.index == 3 and tr.first_convergence.block.end_ns == 12.5 + 2.5 * 4
    assert tr.after_convergence() == {"n_transitions_after_first_convergence": 5,
                                      "n_unstable_transitions_after_first_convergence": 1,
                                      "remained_stable_after_first_convergence": False}
    assert transitions_table(tr)["convergence_reached"].iloc[-1] and tr.stable_in_a_row == 3


def test_not_converged_reports_none():
    tr = _run([_fp(1.15 ** b) for b in range(7)])
    assert tr.first_convergence is None
    assert tr.after_convergence()["remained_stable_after_first_convergence"] is None


def test_transitions_table_has_the_requested_columns():
    tr = _run([_fp()] * 3)
    assert list(transitions_table(tr).columns) == [
        "transition", "previous_block", "block", "block_start_ns", "block_end_ns",
        "FP_relative_change", "FP_pass", "RDF_nRMSD", "RDF_pass", "Spearman_rho",
        "Spearman_pass", "all_criteria_pass", "stable_transition_count", "convergence_reached"]
