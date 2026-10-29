"""Tests for the shared `convergence` module: one transition's stability, and convergence over a run."""
import dataclasses

import numpy as np
import pytest
import xarray as xr

from oneopes.errors import InputError
from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    ConvergenceSummary,
    StabilityResult,
    Tolerances,
    check_stability,
    compare_blocks,
    first_convergence,
    nrmsd,
    stable_in_a_row,
    summarize,
)
from oneopes.trajectory_blocks import block_measurement, stack_blocks

TOL = DEFAULT_TOLERANCES


def _same_g(n_atoms):
    """Identical g(r) for every atom (and, reused across blocks, nRMSD = 0)."""
    return np.tile([0.0, 0.5, 1.0, 1.0, 1.0], (n_atoms, 1))


def _block(fp, g):
    """A block's hydration: only the FP and g(r) the criteria look at."""
    return xr.Dataset({"fp": ("atom", np.asarray(fp, dtype=float)),
                       "g": (("atom", "r_nm"), np.asarray(g, dtype=float))})


def _stack(blocks, block_ns=1.0, start_ns=0.0):
    """The blocks as consecutive one-frame blocks of block_ns ns from start_ns."""
    return stack_blocks([block_measurement(block, range(b, b + 1), (start_ns + b * block_ns,
                                                                    start_ns + (b + 1) * block_ns))
                         for b, block in enumerate(blocks)])


def _fp_run(*fps):
    """Consecutive blocks with the given FP values and identical g(r)."""
    return _stack([_block(fp, _same_g(len(fp))) for fp in fps])


def _n_blocks(n, block_ns=1.0):
    """n consecutive blocks of block_ns ns from 0, all alike."""
    return _stack([_block([-1.0], _same_g(1))] * n, block_ns)


def _stabilities(passed):
    """Stability results that only differ in whether they passed."""
    return [StabilityResult(0.0, 0.0, 1.0, p, p, p, p) for p in passed]


# --- one transition ---------------------------------------------------------

def test_default_tolerances():
    assert (TOL.fp_rel_tol, TOL.rdf_nrmsd_tol, TOL.spearman_tol, TOL.n_stable) == (0.10, 0.15, 0.90, 3)


@pytest.mark.parametrize("bad", [{"fp_rel_tol": -0.1}, {"rdf_nrmsd_tol": -1.0}, {"spearman_tol": 1.5},
                                 {"spearman_tol": -2.0}, {"n_stable": 0}])
def test_tolerances_reject_values_that_mean_nothing(bad):
    with pytest.raises(InputError):
        Tolerances(**bad)


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
    per_atom = nrmsd(_block([0.0, 0.0], g).g, _block([0.0, 0.0], prev_g).g)
    assert per_atom.dims == ("atom",)
    assert per_atom[0] == 0.0
    assert per_atom[1] == pytest.approx(np.sqrt(0.5))  # changes 1, .5, 0, -.5, -1 over a peak of 1


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


def test_summarize_converged_run():
    """Converged at transition 2->3 (block 3); of the two later transitions one failed."""
    stabilities = _stabilities([True, True, True, False, True])
    summary = summarize(stabilities, _n_blocks(6, block_ns=2.0), 2.0, TOL)
    assert summary == ConvergenceSummary(block_ns=2.0, tolerances=TOL, n_blocks=6, converged=True,
                                         converged_block=3, converged_ns=8.0, unstable_after_convergence=1,
                                         stable_in_a_row_at_end=1)


def test_summarize_unconverged_and_single_block_runs():
    summary = summarize(_stabilities([True, False]), _n_blocks(3), 1.0, TOL)
    assert (summary.converged, summary.converged_block, summary.converged_ns) == (False, None, None)
    assert summarize([], _n_blocks(1), 1.0, TOL).stable_in_a_row_at_end == 0
