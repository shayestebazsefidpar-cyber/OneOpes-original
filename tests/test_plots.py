"""Charts of a block run: one panel per atom, in the run's atom order."""
import numpy as np
import xarray as xr

from oneopes.block_run import BlockOptions, block_run
from oneopes.convergence import UNCHECKED, FpPrecision
from oneopes.plots import fp_chart, g_chart, write_charts
from oneopes.trajectory_blocks import stack_blocks
from oneopes.waterfp import HydrationOptions, RadialGrid
from synthetic_blocks import one_frame_blocks

ATOMS = ["O1", "C1"]


def _check(passed: bool) -> xr.Dataset:
    """A check that found production from block 1 (1 ns) and passed, or failed only its goal."""
    found = {"production_start_block": 1, "production_start_ns": 1.0, "n_production_blocks": 2, "max_drift_z": 0.0,
             "drift_limit": 3.0, "goal_score": 0.0, "enough_blocks": True, "no_drift": True}
    return xr.Dataset(found | {"goal_met": passed, "passed": passed})


def _run(n_blocks: int, passed: bool, fps: list[list[float]] | None = None) -> xr.Dataset:
    """Blocks with a flat g(r) and, unless given, the same FP; every check after the first block passes, or none.

    With checks passing, the run converges at block 2, the second in a row (the default n_passes).
    """
    r_nm = RadialGrid().r_nm
    fps = fps or [[-3.0, -1.0]] * n_blocks
    measured = [xr.Dataset({"fp": ("atom", fp), "g": (("atom", "r_nm"), np.ones((len(ATOMS), r_nm.size)))},
                           coords={"atom": ATOMS, "r_nm": r_nm}) for fp in fps]
    blocks = stack_blocks([block.label(fp) for block, fp in zip(one_frame_blocks(n_blocks), measured, strict=True)])
    checks = [UNCHECKED] + [_check(passed)] * (n_blocks - 1)
    return block_run(blocks, checks, BlockOptions(block_ns=1.0), HydrationOptions(), FpPrecision())


def test_fp_is_drawn_as_a_change_from_each_atom_s_mean_on_one_shared_scale():
    """Atoms with FPs -4 and -2, then -2 and -1: each is 33% off its mean, either way."""
    chart = fp_chart(_run(2, passed=True, fps=[[-4.0, -2.0], [-2.0, -1.0]]))
    assert chart.data["fp_change_pct"].round(1).tolist() == [-33.3, -33.3, 33.3, 33.3]
    assert "scale" not in chart.to_dict().get("resolve", {})


def test_panels_follow_the_run_s_atom_order():
    chart = fp_chart(_run(3, passed=True)).to_dict()
    assert chart["facet"]["sort"] == ATOMS


def test_lines_mark_where_production_starts_and_where_the_run_converged():
    assert len(fp_chart(_run(3, passed=True)).to_dict()["spec"]["layer"]) == 3
    assert len(fp_chart(_run(3, passed=False)).to_dict()["spec"]["layer"]) == 2
    assert len(fp_chart(_run(1, passed=False)).to_dict()["spec"]["layer"]) == 1


def test_g_is_drawn_up_to_1_nm_with_about_100_points_per_curve():
    data = g_chart(_run(2, passed=True)).data
    curve = data[(data.atom == "O1") & (data.block == 0)]
    assert curve.r_nm.max() <= 1.0 and 90 <= len(curve) <= 110


def test_the_charts_are_written_as_two_pngs(tmp_path):
    write_charts(_run(3, passed=True), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fp_vs_time.png", "g_per_block.png"]
