"""Convergence monitor of an unbiased ligand-in-water run (``oneopes monitor``)."""
import logging
import os
import time
import signal
from collections.abc import Callable, Iterator, Sequence
from itertools import count
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from cyclopts import Parameter
from cyclopts.types import ExistingFile

from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    StabilityResult,
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

logger = logging.getLogger(__name__)


def process_alive(pid: int | None) -> bool:
    """Whether process ``pid`` is running (False without a pid)."""
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


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


def stop_mdrun(pid: int | None) -> None:
    """Send SIGTERM to mdrun, if it is still running (never SIGKILL: see docs/ligand_setup.md)."""
    if pid is None or not process_alive(pid):
        return
    logger.info("sending SIGTERM to mdrun pid %d", pid)
    try:
        os.kill(pid, signal.SIGTERM)
    except OSError as e:
        logger.warning("failed to signal mdrun pid %d: %s", pid, e)


def log_progress(block: int, end_ns: float, stabilities: Sequence[StabilityResult]) -> None:
    """One log line for the transition into ``block``, the last of ``stabilities``."""
    if not stabilities:
        return
    s = stabilities[-1]
    logger.info("block %d (t=%.1f ns): max|dFP/FP|=%.3f max_nRMSD_RDF=%.3f spearman=%.3f stable=%s streak=%d",
                block, end_ns, s.fp_change, s.rdf_nrmsd, s.spearman_rho, s.passed, stable_in_a_row(stabilities)[-1])


def plot_blocks(blocks: Sequence[WaterFP], block_times: Sequence[tuple[float, float]], outdir: Path) -> None:
    """For every ligand atom, FP against time (fp_plots/) and g(r) of each block (rdf_plots/)."""
    (outdir / "fp_plots").mkdir(parents=True, exist_ok=True)
    (outdir / "rdf_plots").mkdir(parents=True, exist_ok=True)
    fp_df, rdf_df = fp_per_block(blocks, block_times), rdf_per_block(blocks)
    for atom_name in fp_df["atom"].unique():
        plot_atom(atom_name, fp_df, rdf_df, outdir)


def select_g1_g2_on_block(block: WaterFP, selection_dir: Path, fp_round_decimals: int) -> None:
    """Write the block's rdf.csv and fingerprints.csv, then its G1/G2 selection as g1_g2.yaml (if there is one)."""
    write_waterfp_files(block, selection_dir)
    try:
        selection = select_g1_g2(block.solute, block.results.fp, fp_round_decimals)
    except ValueError as e:
        logger.warning("no G1/G2 selection: %s", e)
        return
    write_g1_g2_yaml(selection, selection_dir / "g1_g2.yaml")
    logger.info("%s, written to %s", selection, selection_dir)


def plot_atom(atom_name: str, fp_df: pd.DataFrame, rdf_df: pd.DataFrame, outdir: Path) -> tuple[Path, Path]:
    """Save one atom's FP against time and its g(r) of each block; return the two paths.

    ``fp_df`` and ``rdf_df`` are the per-block tables of :func:`fp_per_block` and :func:`rdf_per_block`;
    the plots go to fp_plots/ and rdf_plots/ in ``outdir``, which must exist.
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
    def load_universe() -> mda.Universe:
        return mda.Universe(str(tpr), str(xtc))

    u = load_universe()
    try:
        sel = select_system(u, ligand_selection, water_selection)
    except SystemDefinitionError as e:
        raise SystemExit(f"[monitor] {e}") from e
    atom_names = list(u.atoms[sel.ligand_heavy].names)
    logger.info("%d ligand heavy atoms (%s): %s", len(atom_names), sel.ligand_description, atom_names)
    logger.info("%d water oxygens (%s)", len(sel.water_oxygens), sel.water_description)
    logger.info("RDF: rmax=%s nm, bins=%d, norm=mean of last %d bins (WaterFP fp.py convention)",
                grid.rmax_nm, grid.n_bins, norm_tail_bins)

    try:
        dt_ps = frame_step_ps(wait_for_frames(load_universe, 2, mdrun_pid, poll_seconds))
        block_length = frames_per_block(block_ns, dt_ps)
    except ValueError as e:
        raise SystemExit(f"[monitor] {e}") from e
    logger.info("dt/frame = %s ps -> %d frames/block (%s ns)", dt_ps, block_length, block_ns)

    outdir.mkdir(parents=True, exist_ok=True)
    blocks, block_times, stabilities = [], [], []
    for u, frames in completed_blocks(load_universe, block_length, mdrun_pid, poll_seconds):
        waterfp = WaterFP(u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens], grid=grid,
                          norm_tail_bins=norm_tail_bins)
        blocks.append(waterfp.run(start=frames.start, stop=frames.stop))
        block_times.append(time_span_ns(u, frames, dt_ps))
        write_block_tables(outdir, blocks, block_times, tol)
        stabilities = compare_blocks([block.results for block in blocks], tol)
        log_progress(len(blocks) - 1, block_times[-1][1], stabilities)
        if first_convergence(stabilities, tol.n_stable) is not None:
            break
    if not blocks:
        raise SystemExit("[monitor] mdrun ended before the first block was complete; nothing to report.")

    summary = summarize(stabilities, block_times, block_ns, tol)
    trajectory_end_ns = frame_time_ns(load_universe(), -1)
    write_summary(outdir / "convergence_summary.json", summary, trajectory_end_ns=trajectory_end_ns,
                  ligand_selection=sel.ligand_description, ligand_heavy_atoms=atom_names)
    plot_blocks(blocks, block_times, outdir)
    converged_block = summary.converged_block
    if converged_block is None:
        logger.info("DONE, not converged: the trajectory ended at %s ns, after %d complete block(s)",
                    trajectory_end_ns, summary.n_blocks)
        return

    logger.info("DONE, converged at block %d (%s ns)", converged_block, summary.converged_ns)
    stop_mdrun(mdrun_pid)
    select_g1_g2_on_block(blocks[converged_block], outdir / "selection", fp_round_decimals)
