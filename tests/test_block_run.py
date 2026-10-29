"""Tests for the `block_run` module: one Dataset per run, its settings and facts, and blocks.nc."""
import numpy as np
import pytest
import xarray as xr

from oneopes.block_run import (
    BlockOptions,
    block_run,
    load_run,
    run_block,
    run_block_options,
    run_facts,
    run_summary,
    save_run,
)
from oneopes.convergence import Tolerances
from oneopes.errors import InputError
from oneopes.trajectory_blocks import Block
from oneopes.waterfp import HydrationOptions

TOL = Tolerances(fp_rel_tol=0.2, rdf_nrmsd_tol=0.15, spearman_tol=0.9, n_stable=2)
OPTIONS = BlockOptions(block_ns=2.0, tolerances=TOL)


def _measured(*fps, block_ns=2.0):
    """One-frame blocks of block_ns ns with the given FP values and the same g(r) for every atom."""
    g = np.tile([0.0, 0.5, 1.0, 1.0], (len(fps[0]), 1))
    return [Block(range(b, b + 1), b * block_ns, (b + 1) * block_ns).label(
                xr.Dataset({"fp": ("atom", np.asarray(fp, dtype=float)), "g": (("atom", "r_nm"), g)},
                           coords={"atom": ["C1", "N1", "O1"][:len(fp)]}))
            for b, fp in enumerate(fps)]


def _run(**facts):
    fp = [-9.0, -5.0, -2.0]
    return block_run(_measured(fp, fp, fp, [-20.0, -5.0, -2.0]), OPTIONS, HydrationOptions(), **facts)


def test_a_run_holds_the_measurements_the_comparisons_and_the_settings():
    run = _run(ligand_selection="resname MOL")
    assert run.sizes["block"] == 4 and {"fp", "g", "passed", "stable_in_a_row"} <= set(run.data_vars)
    assert run_block_options(run) == OPTIONS
    assert (run.attrs["rmax_nm"], run.attrs["norm_tail_bins"]) == (2.001, 500)
    assert run_facts(run) == {"ligand_selection": "resname MOL"}


def test_the_summary_is_derived_from_the_run():
    summary = run_summary(_run())
    assert (summary.converged_block, summary.converged_ns, summary.block_ns) == (2, 6.0, 2.0)
    assert (summary.n_blocks, summary.unstable_after_convergence, summary.tolerances) == (4, 1, TOL)


def test_run_block_defaults_to_the_converged_block():
    run = _run()
    assert int(run_block(run).block) == 2
    assert int(run_block(run, 0).block) == 0
    with pytest.raises(InputError, match="blocks 0 to 3, not 4"):
        run_block(run, 4)


def test_run_block_of_an_unconverged_run_needs_a_block():
    fp = [-9.0, -5.0, -2.0]
    run = block_run(_measured(fp, [-20.0, -5.0, -2.0]), OPTIONS, HydrationOptions())
    with pytest.raises(InputError, match="did not converge"):
        run_block(run)


def test_a_saved_run_loads_back_identical(tmp_path):
    run = _run(ligand_selection="resname MOL", ligand_heavy_atoms=["C1", "N1", "O1"], trajectory_end_ns=8.5)
    save_run(run, tmp_path / "blocks.nc")
    loaded = load_run(tmp_path / "blocks.nc")
    assert loaded.identical(run)
    assert run_summary(loaded) == run_summary(run)
    assert run_facts(loaded) == {"ligand_selection": "resname MOL", "ligand_heavy_atoms": ["C1", "N1", "O1"],
                                 "trajectory_end_ns": 8.5}
