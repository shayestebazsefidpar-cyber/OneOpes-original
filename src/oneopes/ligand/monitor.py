"""Convergence monitor of an unbiased ligand-in-water run (``oneopes monitor``)."""
import os
import time
import signal
from collections.abc import Callable, Iterator
from itertools import count
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from cyclopts import Parameter
from cyclopts.types import ExistingFile

from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    Tolerances,
    compare_blocks,
    first_convergence,
    fp_per_block,
    rdf_per_block,
    stable_in_a_row,
    summarize,
    write_block_tables,
    write_summary,
)
from oneopes.ligand.g1_g2 import select_g1_g2, write_g1_g2_yaml, write_waterfp_files
from oneopes.system import SystemDefinitionError, select_system
from oneopes.trajectory_blocks import (
    BLOCK_NS_DEFAULT,
    frame_step_ps,
    frame_time_ns,
    frames_per_block,
    time_span_ns,
)
from oneopes.waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def wait_for_frames(load_universe, n_frames: int, mdrun_pid: int | None, poll_seconds: float) -> mda.Universe:
    """Reload the trajectory until it has ``n_frames`` frames or mdrun is no longer running."""
    while True:
        u = load_universe()
        if len(u.trajectory) >= n_frames or not process_alive(mdrun_pid):
            return u
        time.sleep(poll_seconds)


def completed_blocks(load_universe: Callable[[], mda.Universe], block_length: int, mdrun_pid: int | None,
                     poll_seconds: float) -> Iterator[tuple[mda.Universe, range]]:
    """Each block of ``block_length`` frames, with a universe holding it, as soon as the trajectory has it.

    Ends when mdrun is no longer running and the next block is incomplete.
    """
    for first in count(0, block_length):
        frames = range(first, first + block_length)
        u = wait_for_frames(load_universe, frames.stop, mdrun_pid, poll_seconds)
        if len(u.trajectory) < frames.stop:
            return
        yield u, frames


def plot_atom(atom_name, fp_df, rdf_df, outdir):
    """Save the two per-atom diagnostic plots (FP vs. simulation time, and
    RDF per block) for `atom_name`, exactly as main()'s plotting loop used
    to build them - same data, same styling, same file paths. No
    printing; returns the two paths written so callers (or tests) can
    check them.

    fp_df: all blocks' FP rows (columns include t_mid_ns, atom, FP).
    rdf_df: all blocks' g(r) rows (columns include block, atom, r_nm,
    g_r). outdir may be a str or a Path.
    """
    outdir = Path(outdir)
    fp_plot_path = outdir / "fp_plots" / f"FP_vs_time_{atom_name}.png"
    rdf_plot_path = outdir / "rdf_plots" / f"RDF_per_block_{atom_name}.png"

    sub = fp_df[fp_df["atom"] == atom_name].sort_values("t_mid_ns")
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(sub["t_mid_ns"], sub["FP"], marker="o", color="tab:blue")
    ax.set_xlabel("simulation time (ns)")
    ax.set_ylabel("FP (WaterFP excess-entropy integral)")
    ax.set_title(f"Hydration FP vs time - {atom_name}")
    fig.tight_layout()
    fig.savefig(fp_plot_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    rsub = rdf_df[rdf_df["atom"] == atom_name]
    for b in sorted(rsub["block"].unique()):
        bb = rsub[rsub["block"] == b]
        ax2.plot(bb["r_nm"], bb["g_r"], alpha=0.6, label=f"block {b}")
    ax2.set_xlabel("r (nm) to water O")
    ax2.set_ylabel("g(r)")
    ax2.set_xlim(0, 1.0)
    ax2.set_title(f"RDF per block - {atom_name}")
    if rsub["block"].nunique() <= 8:
        ax2.legend(fontsize=6)
    fig2.tight_layout()
    fig2.savefig(rdf_plot_path, dpi=150)
    plt.close(fig2)

    return fp_plot_path, rdf_plot_path


def monitor(mdrun_pid: int | None = None, /, *, tpr: ExistingFile = Path("prod.tpr"),
            xtc: ExistingFile = Path("prod.xtc"), outdir: Path = Path("outputs/convergence"),
            ligand_selection: str | None = None, water_selection: str | None = None,
            grid: Annotated[RadialGrid, Parameter(name="*")] = RadialGrid(),
            norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT, block_ns: float = BLOCK_NS_DEFAULT,
            tol: Annotated[Tolerances, Parameter(name="*")] = DEFAULT_TOLERANCES,
            poll_seconds: int = 30, fp_round_decimals: int = 1) -> None:
    """Follow an unbiased ligand-in-water run block by block and stop it once its hydration has converged.

    Analyses the trajectory block by block as mdrun writes it, compares each block with the
    previous one, and once --n-stable transitions in a row are stable sends SIGTERM to the
    mdrun PID. Tables, plots and convergence_summary.json go to --outdir; on convergence, the
    G1/G2 selection of the converged block goes to --outdir/selection.
    docs/METHOD_RATIONALE.md explains the criteria.

    Parameters
    ----------
    mdrun_pid
        PID of a running gmx mdrun to SIGTERM on convergence; without it, only report.
    ligand_selection
        MDAnalysis selection of the ligand (default: auto-detect).
    water_selection
        MDAnalysis selection of the water (default: known water residue names).
    norm_tail_bins
        Bins at the end of the profile whose mean is the bulk density.
    block_ns
        Trajectory block size (ns).
    poll_seconds
        Wait between polls for more trajectory frames.
    fp_round_decimals
        Decimals FP is rounded to when the G1/G2 selection compares atoms.
    """
    (outdir / "rdf_plots").mkdir(parents=True, exist_ok=True)
    (outdir / "fp_plots").mkdir(parents=True, exist_ok=True)

    def get_universe():
        return mda.Universe(str(tpr), str(xtc))

    u = get_universe()
    try:
        sel = select_system(u, ligand_selection, water_selection)
    except SystemDefinitionError as e:
        raise SystemExit(f"[monitor] {e}")
    heavy_names = list(u.atoms[sel.ligand_heavy].names)
    print(f"[monitor] {len(heavy_names)} ligand heavy atoms ({sel.ligand_description}): {heavy_names}")
    print(f"[monitor] {len(sel.water_oxygens)} water oxygens ({sel.water_description})")
    print(f"[monitor] RDF: rmax={grid.rmax_nm} nm, bins={grid.n_bins}, "
          f"norm=mean of last {norm_tail_bins} bins (WaterFP fp.py convention)")

    u = wait_for_frames(get_universe, 2, mdrun_pid, poll_seconds)
    try:
        dt_ps = frame_step_ps(u)
        block_length = frames_per_block(block_ns, dt_ps)
    except ValueError as e:
        raise SystemExit(f"[monitor] {e}")
    print(f"[monitor] dt/frame = {dt_ps} ps -> {block_length} frames/block ({block_ns} ns)")

    blocks, block_times, stabilities = [], [], []
    for u, frames in completed_blocks(get_universe, block_length, mdrun_pid, poll_seconds):
        blocks.append(WaterFP(u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens], grid=grid,
                              norm_tail_bins=norm_tail_bins).run(start=frames.start, stop=frames.stop))
        block_times.append(time_span_ns(u, frames, dt_ps))
        write_block_tables(outdir, blocks, block_times, tol)
        stabilities = compare_blocks([block.results for block in blocks], tol)
        if stabilities:
            s = stabilities[-1]
            print(f"[monitor] block {len(blocks) - 1} (t={block_times[-1][1]:.1f} ns): "
                  f"max|dFP/FP|={s.fp_change:.3f} max_nRMSD_RDF={s.rdf_nrmsd:.3f} spearman={s.spearman_rho:.3f} "
                  f"stable={s.passed} streak={stable_in_a_row(stabilities)[-1]}")
        if first_convergence(stabilities, tol.n_stable) is not None:
            break

    if not blocks:
        raise SystemExit("[monitor] mdrun ended before the first block was complete; nothing to report.")
    summary = summarize(stabilities, block_times, block_ns, tol)
    trajectory_end_ns = frame_time_ns(get_universe(), -1)
    write_summary(outdir / "convergence_summary.json", summary, trajectory_end_ns=trajectory_end_ns,
                  ligand_selection=sel.ligand_description, ligand_heavy_atoms=heavy_names)

    fp_df = fp_per_block(blocks, block_times)
    rdf_df = rdf_per_block(blocks)
    for name in heavy_names:
        plot_atom(name, fp_df, rdf_df, outdir)

    if not summary.converged:
        print(f"[monitor] DONE, not converged: the trajectory ended at {trajectory_end_ns} ns, "
              f"after {summary.n_blocks} complete block(s)")
        return
    print(f"[monitor] DONE, converged at block {summary.converged_block} ({summary.converged_ns} ns)")

    if process_alive(mdrun_pid):
        print(f"[monitor] sending SIGTERM to mdrun pid {mdrun_pid}")
        try:
            os.kill(mdrun_pid, signal.SIGTERM)
        except OSError as e:
            print(f"[monitor] failed to signal mdrun pid {mdrun_pid}: {e}")

    block, selection_dir = blocks[summary.converged_block], outdir / "selection"
    write_waterfp_files(block, selection_dir)
    try:
        selection = select_g1_g2(block.solute, block.results.fp, fp_round_decimals)
    except ValueError as e:
        print(f"[monitor] no G1/G2 selection for block {summary.converged_block}: {e}")
        return
    write_g1_g2_yaml(selection, selection_dir / "g1_g2.yaml")
    print(f"[monitor] {selection}, written to {selection_dir}")
