"""Tests for the `convergence` module: production, the jackknife FP, drift, and the checks of a run."""
import dataclasses

import numpy as np
import pytest
import xarray as xr
from scipy.stats import t as student_t

from oneopes.convergence import (
    DEFAULT_CRITERIA,
    UNCHECKED,
    ConvergenceCriteria,
    ConvergenceSummary,
    FpPrecision,
    check,
    check_every_block,
    combined,
    converged_block,
    drift_limit,
    fp_samples,
    jackknife,
    passes_in_a_row,
    production,
    production_estimate,
    production_start,
    summarize,
    with_checks,
)
from oneopes.errors import InputError
from synthetic_blocks import COARSE_OPTIONS, water_blocks

DEPLETED = 0.5
"""A first shell at half the bulk density: a clearly negative FP."""
LOOSE_GOAL = FpPrecision(rel_se_tol=1.0)
UNREACHABLE_GOAL = FpPrecision(rel_se_tol=0.0)


# --- criteria -----------------------------------------------------------------

def test_default_criteria():
    assert dataclasses.astuple(DEFAULT_CRITERIA) == (10, 0.05, 2)


@pytest.mark.parametrize("bad", [{"min_blocks": 3}, {"drift_alpha": -0.1}, {"drift_alpha": 1.5}, {"n_passes": 0}])
def test_criteria_reject_values_that_mean_nothing(bad):
    with pytest.raises(InputError):
        ConvergenceCriteria(**bad)


# --- production ---------------------------------------------------------------

def test_production_starts_where_the_first_shell_has_equilibrated_or_a_block_later():
    assert production_start(water_blocks([2.0] * 4 + [1.0] * 8).shell_count) in (4, 5)


def test_a_stationary_run_is_production_from_the_start():
    assert production_start(water_blocks([1.0] * 8).shell_count) == 0


def test_production_keeps_at_least_the_last_two_blocks():
    blocks = water_blocks([2.0] * 5 + [1.0])
    assert production_start(blocks.shell_count) == 5
    assert list(production(blocks).block.values) == [4, 5]


# --- jackknife ----------------------------------------------------------------

def test_the_jackknife_removes_the_finite_sampling_bias_of_fp():
    """Bulk water has FP 0: noise alone makes the uncorrected FP negative; the corrected one is 0 within its error."""
    estimate = jackknife(water_blocks([1.0] * 12), COARSE_OPTIONS)
    assert (estimate.fp_uncorrected < -3 * estimate.fp_se).all()
    assert (abs(estimate.fp) < 3 * estimate.fp_se).all()


def test_each_left_out_fp_is_the_fp_of_the_other_blocks():
    blocks = water_blocks([DEPLETED] * 4)
    left_out = jackknife(blocks, COARSE_OPTIONS).fp_left_out
    for j in range(4):
        np.testing.assert_allclose(left_out.isel(block=j), combined(blocks.drop_isel(block=j), COARSE_OPTIONS).fp)


def test_identical_blocks_have_no_error_and_no_correction():
    blocks = xr.concat([water_blocks([DEPLETED]).isel(block=0)] * 3, dim="block")
    estimate = jackknife(blocks, COARSE_OPTIONS)
    np.testing.assert_allclose(estimate.fp, estimate.fp_uncorrected)
    np.testing.assert_allclose(estimate.fp_se, 0.0, atol=1e-12)


# --- drift --------------------------------------------------------------------

def test_a_change_between_the_halves_of_production_is_drift():
    estimate = production_estimate(water_blocks([DEPLETED] * 6 + [1.0] * 6), COARSE_OPTIONS)
    assert (estimate.drift_z > drift_limit(2, 12, 0.05)).all()


def test_a_stationary_run_has_no_drift():
    estimate = production_estimate(water_blocks([DEPLETED] * 12), COARSE_OPTIONS)
    assert (estimate.drift_z < drift_limit(2, 12, 0.05)).all()


def test_a_step_between_the_halves_is_drift_however_many_blocks_share_it():
    """The error of each half comes from that half alone, so a step does not hide itself by widening the error."""
    estimate = production_estimate(water_blocks([DEPLETED] * 4 + [1.0] * 4), COARSE_OPTIONS)
    assert (estimate.drift_z > drift_limit(2, 8, 0.05)).all()


def test_drift_needs_two_blocks_in_each_half():
    assert np.isnan(production_estimate(water_blocks([DEPLETED] * 3), COARSE_OPTIONS).drift_z).all()


def test_the_drift_limit_is_student_s_t_corrected_for_the_number_of_atoms():
    assert drift_limit(1, 10, 0.05) == pytest.approx(student_t.ppf(0.975, df=8))
    assert drift_limit(100, 10, 0.05) == pytest.approx(student_t.ppf(1 - 0.05 / 200, df=8))
    assert drift_limit(5, 10, 0.0) == np.inf


# --- FP draws and goals ---------------------------------------------------------

def test_fp_draws_centre_on_the_fp_and_spread_as_its_errors():
    estimate = jackknife(water_blocks([DEPLETED] * 10, n_atoms=3), COARSE_OPTIONS)
    draws = fp_samples(estimate, 20000, np.random.default_rng(1))
    np.testing.assert_allclose(draws.std(axis=0), estimate.fp_se, rtol=0.05)
    np.testing.assert_allclose(draws.mean(axis=0), estimate.fp, atol=0.05 * float(estimate.fp_se.max()))


def test_fp_precision_scores_the_largest_error_relative_to_its_fp():
    goal = FpPrecision(rel_se_tol=0.05)
    score = goal.score(xr.Dataset({"fp": ("atom", [-10.0, -2.0]), "fp_se": ("atom", [0.1, 0.2])}))
    assert score == pytest.approx(0.1)
    assert not goal.met(score) and goal.met(0.05)


# --- checks -------------------------------------------------------------------

def test_a_single_block_is_not_checked():
    unchecked = check(water_blocks([DEPLETED]), DEFAULT_CRITERIA, COARSE_OPTIONS, LOOSE_GOAL)
    assert np.isnan(unchecked.production_start_block) and not unchecked.passed


def test_checks_pass_from_the_block_completing_min_blocks_of_production():
    blocks = water_blocks([DEPLETED] * 6)
    checks = check_every_block(blocks, ConvergenceCriteria(min_blocks=4), COARSE_OPTIONS, LOOSE_GOAL)
    assert [bool(c.passed) for c in checks] == [False, False, False, True, True, True]
    assert [int(c.n_production_blocks) for c in checks[1:]] == [2, 3, 4, 5, 6]


def test_drift_fails_a_check():
    """The FP drifts while the first-shell counts, which production is found from, do not."""
    steady = water_blocks([DEPLETED] * 8, seed=1).shell_count
    blocks = water_blocks([DEPLETED] * 4 + [1.0] * 4).assign(shell_count=steady)
    result = check(blocks, ConvergenceCriteria(min_blocks=4), COARSE_OPTIONS, LOOSE_GOAL)
    assert (result.enough_blocks, result.no_drift, result.passed) == (True, False, False)
    assert result.max_drift_z > result.drift_limit


def test_an_unmet_goal_fails_a_check():
    result = check(water_blocks([DEPLETED] * 4), ConvergenceCriteria(min_blocks=4), COARSE_OPTIONS, UNREACHABLE_GOAL)
    assert (result.enough_blocks, result.no_drift, result.goal_met, result.passed) == (True, True, False, False)


# --- a run of checks -------------------------------------------------------------

def _checked_run(passed, block_ns=1.0):
    """A run of blocks whose checks only differ in whether they passed; block 0 is unchecked."""
    blocks = xr.Dataset(coords={"block": np.arange(len(passed) + 1),
                                "t_end_ns": ("block", (np.arange(len(passed) + 1) + 1) * block_ns)})
    found = {"production_start_block": 1, "production_start_ns": block_ns, "n_production_blocks": 3,
             "max_drift_z": 0.0, "drift_limit": 3.0, "goal_score": 1.0, "enough_blocks": True, "no_drift": True}
    checks = [UNCHECKED, *(xr.Dataset(found | {"goal_met": passes, "passed": passes}) for passes in passed)]
    return with_checks(blocks, checks)


def test_passes_in_a_row_resets_on_a_failed_check():
    passed = [False] * 4 + [True] * 3 + [False] + [True] * 2
    assert passes_in_a_row(passed) == [0, 0, 0, 0, 1, 2, 3, 0, 1, 2]


def test_converged_block_completes_the_first_run_of_passes():
    run = _checked_run([True, True, False, True, True, True, False, True, True, True])
    assert converged_block(run, 3) == 6
    assert converged_block(run.isel(block=slice(0, 6)), 3) is None


def test_summarize_a_converged_run():
    """Converged at block 3, the second passing check in a row; one later check failed."""
    summary = summarize(_checked_run([False, True, True, False, True], block_ns=2.0), 2.0, DEFAULT_CRITERIA)
    assert summary == ConvergenceSummary(block_ns=2.0, criteria=DEFAULT_CRITERIA, n_blocks=6, converged=True,
                                         converged_block=3, converged_ns=8.0, failed_after_convergence=1,
                                         passes_in_a_row_at_end=1, production_start_ns=2.0, n_production_blocks=3)


def test_summarize_an_unconverged_run_and_one_of_a_single_block():
    summary = summarize(_checked_run([True, False]), 1.0, DEFAULT_CRITERIA)
    assert (summary.converged, summary.converged_block, summary.converged_ns) == (False, None, None)
    single = summarize(_checked_run([]), 1.0, DEFAULT_CRITERIA)
    assert (single.passes_in_a_row_at_end, single.production_start_ns, single.n_production_blocks) == (0, None, None)
