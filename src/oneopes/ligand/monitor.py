"""Convergence monitor of an unbiased ligand-in-water run (``oneopes monitor``)."""
import logging
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
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number

from oneopes.block_run import BlockOptions, block_run, run_block, run_summary
from oneopes.convergence import converged_block
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import write_selection
from oneopes.reports import write_run
from oneopes.system import SystemOptions, select_system
from oneopes.trajectory_blocks import block_of, frame_step_ps, frame_time_ns, frames_per_block, label_block
from oneopes.waterfp import HydrationOptions, measure_hydration

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


def log_progress(blocks: xr.Dataset) -> None:
    """One log line for the latest block of a :func:`block_run` (none for block 0)."""
    if blocks.sizes["block"] < 2:
        return
    latest = blocks.isel(block=-1)
    logger.info("block %d (t=%.1f ns): max|dFP/FP|=%.3f max_nRMSD_RDF=%.3f spearman=%.3f stable=%s streak=%d",
                latest.block, latest.t_end_ns, latest.fp_change, latest.rdf_nrmsd, latest.spearman_rho,
                bool(latest.passed), latest.stable_in_a_row)


def plot_blocks(blocks: xr.Dataset, outdir: Path) -> None:
    """For every ligand atom, FP against time (fp_plots/) and g(r) of each block (rdf_plots/)."""
    (outdir / "fp_plots").mkdir(parents=True, exist_ok=True)
    (outdir / "rdf_plots").mkdir(parents=True, exist_ok=True)
    for atom_name in blocks.atom.values:
        plot_atom(str(atom_name), blocks, outdir)


def plot_atom(atom_name: str, blocks: xr.Dataset, outdir: Path) -> tuple[Path, Path]:
    """Save one atom's FP against time and its g(r) of each block; return the two paths.

    ``blocks`` are the stacked block measurements; the plots go to fp_plots/ and rdf_plots/
    in ``outdir``, which must exist.
    """
    outdir = Path(outdir)
    fp_plot_path = outdir / "fp_plots" / f"FP_vs_time_{atom_name}.png"
    rdf_plot_path = outdir / "rdf_plots" / f"RDF_per_block_{atom_name}.png"
    atom = blocks.sel(atom=atom_name)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(atom.t_mid_ns, atom.fp, marker="o", color="tab:blue")
    ax.set_xlabel("simulation time (ns)")
    ax.set_ylabel("FP (WaterFP excess-entropy integral)")
    ax.set_title(f"Hydration FP vs time - {atom_name}")
    fig.tight_layout()
    fig.savefig(fp_plot_path, dpi=150)
    plt.close(fig)

    fig2, ax2 = plt.subplots(figsize=(5, 3.5))
    for block in atom.block.values:
        ax2.plot(atom.r_nm, atom.g.sel(block=block), alpha=0.6, label=f"block {block}")
    ax2.set_xlabel("r (nm) to water O")
    ax2.set_ylabel("g(r)")
    ax2.set_xlim(0, 1.0)
    ax2.set_title(f"RDF per block - {atom_name}")
    if atom.sizes["block"] <= 8:
        ax2.legend(fontsize=6)
    fig2.tight_layout()
    fig2.savefig(rdf_plot_path, dpi=150)
    plt.close(fig2)

    return fp_plot_path, rdf_plot_path


def monitor(mdrun_pid: int | None = None, /, *, tpr: ExistingFile = Path("prod.tpr"),
            xtc: ExistingFile = Path("prod.xtc"), outdir: Path = Path("outputs/convergence"),
            system: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions(),
            hydration: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions(),
            blocks: Annotated[BlockOptions, Parameter(name="*")] = BlockOptions(),
            poll_seconds: Annotated[float, Parameter(validator=Number(gt=0))] = 30,
            fp_round_decimals: Annotated[int, Parameter(validator=Number(gte=0))] = 1) -> None:
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
    poll_seconds
        Wait between polls for more trajectory frames.
    fp_round_decimals
        Decimals FP is rounded to when the G1/G2 selection compares atoms.
    """
    def load_universe() -> mda.Universe:
        return mda.Universe(str(tpr), str(xtc))

    u = load_universe()
    sel = select_system(u, system)
    ligand = u.atoms[sel.ligand_heavy]
    atom_names = list(ligand.names)
    logger.info("%d ligand heavy atoms (%s): %s", len(atom_names), sel.ligand_selection, atom_names)
    logger.info("%d water oxygens (%s)", len(sel.water_oxygens), sel.water_selection)
    logger.info("RDF: rmax=%s nm, bins=%d, norm=mean of last %d bins (WaterFP fp.py convention)",
                hydration.grid.rmax_nm, hydration.grid.n_bins, hydration.norm_tail_bins)

    dt_ps = frame_step_ps(wait_for_frames(load_universe, 2, mdrun_pid, poll_seconds))
    block_length = frames_per_block(blocks.block_ns, dt_ps)
    logger.info("dt/frame = %s ps -> %d frames/block (%s ns)", dt_ps, block_length, blocks.block_ns)

    outdir.mkdir(parents=True, exist_ok=True)
    measured, run = [], None
    for u, frames in completed_blocks(load_universe, block_length, mdrun_pid, poll_seconds):
        block = block_of(u, frames, dt_ps)
        measurement = measure_hydration(u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens], block.frames,
                                        hydration)
        measured.append(label_block(measurement, block))
        run = block_run(measured, blocks, hydration, ligand_selection=sel.ligand_selection,
                        ligand_heavy_atoms=atom_names)
        write_run(outdir, run)
        log_progress(run)
        if converged_block(run, blocks.tolerances.n_stable) is not None:
            break
    if run is None:
        raise InputError("mdrun ended before the first block was complete; nothing to report.")

    trajectory_end_ns = frame_time_ns(load_universe(), -1)
    run = run.assign_attrs(trajectory_end_ns=trajectory_end_ns)
    write_run(outdir, run)
    summary = run_summary(run)
    plot_blocks(run, outdir)
    if summary.converged_block is None:
        logger.info("DONE, not converged: the trajectory ended at %s ns, after %d complete block(s)",
                    trajectory_end_ns, summary.n_blocks)
        return

    logger.info("DONE, converged at block %d (%s ns)", summary.converged_block, summary.converged_ns)
    stop_mdrun(mdrun_pid)
    try:
        write_selection(ligand, run_block(run), outdir / "selection", fp_round_decimals)
    except InputError as e:
        logger.warning("no G1/G2 selection: %s", e)
