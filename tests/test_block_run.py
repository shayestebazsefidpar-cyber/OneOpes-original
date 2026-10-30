"""Tests for the `block_run` module: one Dataset per run, its settings and facts, its estimates, and blocks.nc."""
import pytest

from oneopes.block_run import (
    BlockOptions,
    block_options_of,
    block_run,
    chosen_block,
    convergence_summary,
    estimate_at,
    facts_of,
    hydration_options_of,
    load_run,
    save_run,
)
from oneopes.convergence import ConvergenceCriteria, FpPrecision, check_every_block, production_estimate
from oneopes.errors import InputError
from synthetic_blocks import COARSE_OPTIONS, water_blocks

OPTIONS = BlockOptions(block_ns=1.0, criteria=ConvergenceCriteria(min_blocks=4))
LOOSE_GOAL = FpPrecision(rel_se_tol=1.0)


def _run(options=OPTIONS, **facts):
    """Six blocks of a depleted first shell, checked: checks 3, 4 and 5 pass, so the run converges at block 4."""
    blocks = water_blocks([0.5] * 6)
    checks = check_every_block(blocks, options.criteria, COARSE_OPTIONS, LOOSE_GOAL)
    return block_run(blocks, checks, options, COARSE_OPTIONS, LOOSE_GOAL, **facts)


def test_a_run_holds_the_blocks_their_checks_and_the_settings():
    run = _run(ligand_selection="resname MOL")
    assert run.sizes["block"] == 6
    assert {"fp", "density", "shell_count", "passed", "passes_in_a_row"} <= set(run.data_vars)
    assert block_options_of(run) == OPTIONS
    assert hydration_options_of(run) == COARSE_OPTIONS
    assert facts_of(run) == {"goal": LOOSE_GOAL.description, "ligand_selection": "resname MOL"}


def test_the_summary_is_derived_from_the_run():
    summary = convergence_summary(_run())
    assert (summary.converged_block, summary.converged_ns, summary.block_ns) == (4, 5.0, 1.0)
    assert (summary.n_blocks, summary.failed_after_convergence, summary.criteria) == (6, 0, OPTIONS.criteria)


def test_the_chosen_block_defaults_to_the_converged_block():
    run = _run()
    assert chosen_block(run) == 4
    assert chosen_block(run, 0) == 0
    with pytest.raises(InputError, match="blocks 0 to 5, not 6"):
        chosen_block(run, 6)


def test_an_unconverged_run_needs_a_chosen_block():
    run = _run(BlockOptions(criteria=ConvergenceCriteria(min_blocks=10)))
    with pytest.raises(InputError, match="did not converge"):
        chosen_block(run)


def test_the_estimate_at_a_block_is_over_the_production_its_check_found():
    run = _run()
    start = int(run.production_start_block.sel(block=5))
    expected = production_estimate(run.sel(block=slice(start, 5)), COARSE_OPTIONS)
    assert estimate_at(run, 5).fp.equals(expected.fp)
    with pytest.raises(InputError, match="Block 0 has no check"):
        estimate_at(run, 0)


def test_a_saved_run_loads_back_identical(tmp_path):
    run = _run(ligand_selection="resname MOL", ligand_heavy_atoms=["A0", "A1"], trajectory_end_ns=8.5)
    save_run(run, tmp_path / "blocks.nc")
    loaded = load_run(tmp_path / "blocks.nc")
    assert loaded.identical(run)
    assert convergence_summary(loaded) == convergence_summary(run)
    assert facts_of(loaded) == {"goal": LOOSE_GOAL.description, "ligand_selection": "resname MOL",
                                "ligand_heavy_atoms": ["A0", "A1"], "trajectory_end_ns": 8.5}
