"""Convergence monitor of an unbiased ligand-in-water run (``oneopes monitor``)."""
import logging
import time
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import psutil
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number

from oneopes.block_run import BlockOptions, block_run, chosen_block, convergence_summary
from oneopes.convergence import converged_block
from oneopes.errors import InputError
from oneopes.ligand.select import write_selection
from oneopes.plots import write_charts
from oneopes.provenance import provenance
from oneopes.reports import write_run
from oneopes.system import SystemOptions, SystemSelection, load_universe, select_system
from oneopes.trajectory_blocks import Block, frame_step_ps, frame_time_ns, frames_per_block
from oneopes.waterfp import HydrationOptions, measure_hydration

logger = logging.getLogger(__name__)


def mdrun_running(pid: int | None) -> bool:
    """Whether ``pid`` is a running mdrun: not ended (a zombie has), and not a reused PID of another program."""
    if pid is None:
        return False
    try:
        process = psutil.Process(pid)
        return process.status() != psutil.STATUS_ZOMBIE and any("mdrun" in Path(part).name
                                                                for part in process.cmdline())
    except psutil.Error:
        return False


def n_complete_frames(u: mda.Universe, mdrun_pid: int | None) -> int:
    """Frames fully written: while mdrun runs, all but the last, which it may still be writing."""
    n_frames = len(u.trajectory)
    return n_frames - 1 if mdrun_running(mdrun_pid) else n_frames


def wait_for_frames(u: mda.Universe, trajectory: Path, n_frames: int, mdrun_pid: int | None,
                    poll_seconds: float) -> None:
    """Reread ``trajectory`` into ``u`` until it has ``n_frames`` complete frames or mdrun has stopped.

    An open trajectory does not see frames written later, hence the rereading.
    """
    while True:
        u.load_new(str(trajectory))
        if n_complete_frames(u, mdrun_pid) >= n_frames or not mdrun_running(mdrun_pid):
            return
        time.sleep(poll_seconds)


def blocks_as_written(u: mda.Universe, trajectory: Path, block_length: int, mdrun_pid: int | None,
                      poll_seconds: float) -> Iterator[range]:
    """The frames of each block, as soon as they are complete; ends when mdrun stops before the next one."""
    for first in count(0, block_length):
        frames = range(first, first + block_length)
        wait_for_frames(u, trajectory, frames.stop, mdrun_pid, poll_seconds)
        if n_complete_frames(u, mdrun_pid) < frames.stop:
            return
        yield frames


def stop_mdrun(pid: int | None) -> None:
    """Send SIGTERM to mdrun if it is still running (never SIGKILL: see docs/ligand_setup.md)."""
    if pid is None or not mdrun_running(pid):
        return
    logger.info("sending SIGTERM to mdrun pid %d", pid)
    try:
        psutil.Process(pid).terminate()
    except psutil.Error as e:
        logger.warning("failed to signal mdrun pid %d: %s", pid, e)


def log_progress(run: xr.Dataset) -> None:
    """Log the latest transition of a block run."""
    if run.sizes["block"] < 2:
        return
    latest = run.isel(block=-1)
    logger.info("block %d (t=%.1f ns): max|dFP/FP|=%.3f max_nRMSD_RDF=%.3f spearman=%.3f stable=%s streak=%d",
                latest.block, latest.t_end_ns, latest.fp_change, latest.rdf_nrmsd, latest.spearman_rho,
                bool(latest.passed), latest.stable_in_a_row)



def follow_until_converged(u: mda.Universe, xtc: Path, sel: SystemSelection, hydration_options: HydrationOptions,
                           block_options: BlockOptions, outdir: Path, mdrun_pid: int | None,
                           poll_seconds: float) -> xr.Dataset:
    """Measure, compare and write each block as mdrun writes it, until convergence or until mdrun stops.

    Raises InputError when mdrun ends before the first block is complete.
    """
    ligand, water = sel.ligand(u), sel.water(u)
    wait_for_frames(u, xtc, 2, mdrun_pid, poll_seconds)
    dt_ps = frame_step_ps(u, n_complete_frames(u, mdrun_pid))
    block_length = frames_per_block(block_options.block_ns, dt_ps)
    logger.info("dt/frame = %s ps -> %d frames/block (%s ns)", dt_ps, block_length, block_options.block_ns)

    measured, run = [], None
    for frames in blocks_as_written(u, xtc, block_length, mdrun_pid, poll_seconds):
        block = Block.of(u, frames, dt_ps)
        measured.append(block.label(measure_hydration(ligand, water, block.frames, hydration_options)))
        run = block_run(measured, block_options, hydration_options, ligand_selection=sel.ligand_selection,
                        ligand_heavy_atoms=list(ligand.names))
        write_run(outdir, run)
        log_progress(run)
        if converged_block(run, block_options.tolerances.n_stable) is not None:
            break
    if run is None:
        raise InputError("mdrun ended before the first block was complete; nothing to report.")
    return run


def monitor(mdrun_pid: int | None = None, /, *, tpr: ExistingFile = Path("prod.tpr"),
            xtc: ExistingFile = Path("prod.xtc"), outdir: Path = Path("outputs/convergence"),
            system_options: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions(),
            hydration_options: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions(),
            block_options: Annotated[BlockOptions, Parameter(name="*")] = BlockOptions(),
            poll_seconds: Annotated[float, Parameter(validator=Number(gt=0))] = 30,
            fp_round_decimals: Annotated[int, Parameter(validator=Number(gte=0))] = 1) -> None:
    """Follow an unbiased ligand-in-water run block by block and stop it once its hydration has converged.

    After --n-stable stable transitions in a row, sends SIGTERM to mdrun and writes the G1/G2 selection
    of the converged block to --outdir/selection. docs/METHOD_RATIONALE.md explains the criteria.

    Parameters
    ----------
    mdrun_pid
        PID of a running gmx mdrun to SIGTERM on convergence; without it, only report.
    tpr
        Topology of the run (with bonds, for the G1/G2 selection).
    xtc
        Trajectory mdrun is writing, or a finished one.
    outdir
        Directory for blocks.nc, the tables, plots and selection.
    poll_seconds
        Wait between polls for more trajectory frames.
    fp_round_decimals
        Decimals FP is rounded to when the G1/G2 selection compares atoms.
    """
    if mdrun_pid is not None and not mdrun_running(mdrun_pid):
        raise InputError(f"pid {mdrun_pid} is not a running gmx mdrun; for a finished run, leave the pid out.")
    u = load_universe(tpr, xtc)
    sel = select_system(u, system_options)
    ligand = sel.ligand(u)
    logger.info("%d ligand heavy atoms (%s): %s", len(ligand), sel.ligand_selection, list(ligand.names))
    logger.info("%d water oxygens (%s)", len(sel.water_oxygens), sel.water_selection)
    logger.info("RDF: rmax=%s nm, bins=%d, norm=mean of last %d bins (WaterFP fp.py convention)",
                hydration_options.grid.rmax_nm, hydration_options.grid.n_bins, hydration_options.norm_tail_bins)

    outdir.mkdir(parents=True, exist_ok=True)
    run = follow_until_converged(u, xtc, sel, hydration_options, block_options, outdir, mdrun_pid, poll_seconds)
    summary = convergence_summary(run)
    if summary.converged_block is not None:
        logger.info("converged at block %d (%s ns)", summary.converged_block, summary.converged_ns)
        stop_mdrun(mdrun_pid)

    u.load_new(str(xtc))
    trajectory_end_ns = frame_time_ns(u, n_complete_frames(u, mdrun_pid) - 1)
    options = {"system_options": system_options, "hydration_options": hydration_options,
               "block_options": block_options, "mdrun_pid": mdrun_pid, "poll_seconds": poll_seconds,
               "fp_round_decimals": fp_round_decimals}
    metadata = provenance({"tpr": tpr, "xtc": xtc}) | {
        "ligand_selection": sel.ligand_selection, "water_selection": sel.water_selection, "options": options}
    run = run.assign_attrs(trajectory_end_ns=trajectory_end_ns)
    write_run(outdir, run, metadata)
    write_charts(run, outdir)
    if summary.converged_block is None:
        logger.info("DONE, not converged: the trajectory ended at %s ns, after %d complete block(s)",
                    trajectory_end_ns, summary.n_blocks)
        return

    try:
        write_selection(ligand, chosen_block(run), outdir / "selection", fp_round_decimals)
    except InputError as e:
        logger.warning("no G1/G2 selection: %s", e)
