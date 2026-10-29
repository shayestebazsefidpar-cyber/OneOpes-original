"""Charts of a block run: one panel per atom, in the run's atom order."""
import numpy as np
import xarray as xr

from oneopes.block_run import BlockOptions, block_run
from oneopes.convergence import Tolerances
from oneopes.plots import fp_chart, g_chart, write_charts
from oneopes.waterfp import HydrationOptions, RadialGrid
from synthetic_blocks import one_frame_blocks

ATOMS = ["O1", "C1"]


def _run(n_blocks: int, n_stable: int) -> xr.Dataset:
    """Blocks with the same FP and a flat g(r), so every transition is stable."""
    r_nm = RadialGrid().r_nm
    measured = xr.Dataset({"fp": ("atom", [-3.0, -1.0]), "g": (("atom", "r_nm"), np.ones((len(ATOMS), r_nm.size)))},
                          coords={"atom": ATOMS, "r_nm": r_nm})
    blocks = [block.label(measured) for block in one_frame_blocks(n_blocks)]
    return block_run(blocks, BlockOptions(block_ns=1.0, tolerances=Tolerances(n_stable=n_stable)), HydrationOptions())


def test_panels_follow_the_run_s_atom_order():
    chart = fp_chart(_run(3, n_stable=2)).to_dict()
    assert chart["facet"]["sort"] == ATOMS


def test_a_dashed_line_marks_convergence_only_when_converged():
    assert len(fp_chart(_run(3, n_stable=2)).to_dict()["spec"]["layer"]) == 2
    assert len(fp_chart(_run(3, n_stable=5)).to_dict()["spec"]["layer"]) == 1


def test_g_is_drawn_up_to_1_nm_with_about_100_points_per_curve():
    data = g_chart(_run(2, n_stable=2)).data
    curve = data[(data.atom == "O1") & (data.block == 0)]
    assert curve.r_nm.max() <= 1.0 and 90 <= len(curve) <= 110


def test_the_charts_are_written_as_two_pngs(tmp_path):
    write_charts(_run(3, n_stable=2), tmp_path)
    assert sorted(path.name for path in tmp_path.iterdir()) == ["fp_vs_time.png", "g_per_block.png"]
