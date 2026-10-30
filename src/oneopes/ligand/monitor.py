"""Convergence monitor of an unbiased ligand-in-water run (``oneopes monitor``)."""
import logging
import time
from collections.abc import Iterator
from itertools import count
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import numpy as np
import psutil
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number

from oneopes.block_run import BlockOptions, block_run, convergence_summary, estimate_at
from oneopes.convergence import PrecisionGoal, check, converged_block
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import SettledAnchors
from oneopes.ligand.select import write_selection, write_selection_draws
from oneopes.plots import write_charts
from oneopes.provenance import provenance
from oneopes.reports import write_run
from oneopes.system import SystemOptions, SystemSelection, load_universe, select_system
from oneopes.trajectory_blocks import Block, frame_step_ps, frame_time_ns, frames_per_block, stack_blocks
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

    An open trajectory does not see frames written later, hence the rereading. A just-started mdrun
    may not have created the trajectory yet; raises InputError if it stops without creating it.
    """
    while True:
        if trajectory.exists():
            u.load_new(str(trajectory))
            if n_complete_frames(u, mdrun_pid) >= n_frames or not mdrun_running(mdrun_pid):
                return
        elif not mdrun_running(mdrun_pid):
            raise InputError(f"{trajectory} does not exist, and no mdrun is running to write it.")
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
    """Log the check after the latest block of a block run."""
    latest = run.isel(block=-1)
    if np.isnan(latest.production_start_block):
        return
    logger.info("block %d (t=%.1f ns): production from %.1f ns (%d blocks), max drift z=%.2f (limit %.2f), "
                "goal score=%.3f, passed=%s, in a row=%d", latest.block, latest.t_end_ns, latest.production_start_ns,
                latest.n_production_blocks, latest.max_drift_z, latest.drift_limit, latest.goal_score,
                bool(latest.passed), latest.passes_in_a_row)


def follow_until_converged(u: mda.Universe, xtc: Path, sel: SystemSelection, hydration_options: HydrationOptions,
                           block_options: BlockOptions, goal: PrecisionGoal, outdir: Path, mdrun_pid: int | None,
                           poll_seconds: float) -> xr.Dataset:
    """Measure, check and write each block as mdrun writes it, until convergence or until mdrun stops.

    Raises InputError when mdrun ends before the first block is complete.
    """
    ligand, water = sel.ligand(u), sel.water(u)
    wait_for_frames(u, xtc, 2, mdrun_pid, poll_seconds)
    dt_ps = frame_step_ps(u, n_complete_frames(u, mdrun_pid))
    block_length = frames_per_block(block_options.block_ns, dt_ps)
    logger.info("dt/frame = %s ps -> %d frames/block (%s ns)", dt_ps, block_length, block_options.block_ns)

    measured, checks, run = [], [], None
    for frames in blocks_as_written(u, xtc, block_length, mdrun_pid, poll_seconds):
        block = Block.of(u, frames, dt_ps)
        measured.append(block.label(measure_hydration(ligand, water, block.frames, hydration_options)))
        blocks = stack_blocks(measured)
        checks.append(check(blocks, block_options.criteria, hydration_options, goal))
        run = block_run(blocks, checks, block_options, hydration_options, goal,
                        ligand_selection=sel.ligand_selection, ligand_heavy_atoms=list(ligand.names))
        write_run(outdir, run)
        log_progress(run)
        if converged_block(run, block_options.criteria.n_passes) is not None:
            break
    if run is None:
        raise InputError("mdrun ended before the first block was complete; nothing to report.")
    return run


def monitor(mdrun_pid: int | None = None, /, *, tpr: ExistingFile = Path("prod.tpr"),
            xtc: Path = Path("prod.xtc"), outdir: Path = Path("outputs/convergence"),
            system_options: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions(),
            hydration_options: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions(),
            block_options: Annotated[BlockOptions, Parameter(name="*")] = BlockOptions(),
            poll_seconds: Annotated[float, Parameter(validator=Number(gt=0))] = 30,
            fp_round_decimals: Annotated[int, Parameter(validator=Number(gte=0))] = 1,
            anchor_confidence: Annotated[float, Parameter(validator=Number(gte=0, lte=1))] = 0.95) -> None:
    """Follow an unbiased ligand-in-water run block by block and stop it once its G1 and G2 anchors are settled.

    After --n-passes passing checks in a row, sends SIGTERM to mdrun and writes the G1/G2 selection
    of the production blocks to --outdir/selection, with how often each selection comes out of FP
    drawn within the errors (g1_g2_draws.csv). docs/METHOD_RATIONALE.md explains the checks.

    Parameters
    ----------
    mdrun_pid
        PID of a running gmx mdrun to SIGTERM on convergence; without it, only report.
    tpr
        Topology of the run (with bonds, for the G1/G2 selection).
    xtc
        Trajectory mdrun is writing (it may not exist yet), or a finished one.
    outdir
        Directory for blocks.nc, the tables, plots and selection.
    poll_seconds
        Wait between polls for more trajectory frames.
    fp_round_decimals
        Decimals FP is rounded to when the G1/G2 selection compares atoms.
    anchor_confidence
        Fraction of FP draws within the errors that must have the same G1 and G2 anchors.
    """
    if mdrun_pid is not None and not mdrun_running(mdrun_pid):
        raise InputError(f"pid {mdrun_pid} is not a running gmx mdrun; for a finished run, leave the pid out.")
    if not xtc.exists() and mdrun_pid is None:
        raise InputError(f"{xtc} does not exist; to follow an mdrun that will write it, give its pid.")
    u = load_universe(tpr, xtc) if xtc.exists() else load_universe(tpr)
    sel = select_system(u, system_options)
    ligand = sel.ligand(u)
    logger.info("%d ligand heavy atoms (%s): %s", len(ligand), sel.ligand_selection, list(ligand.names))
    logger.info("%d water oxygens (%s)", len(sel.water_oxygens), sel.water_selection)
    logger.info("RDF: rmax=%s nm, bins=%d, norm=mean of last %d bins (WaterFP fp.py convention)",
                hydration_options.grid.rmax_nm, hydration_options.grid.n_bins, hydration_options.norm_tail_bins)

    outdir.mkdir(parents=True, exist_ok=True)
    run = follow_until_converged(u, xtc, sel, hydration_options, block_options, SettledAnchors(anchor_confidence),
                                 outdir, mdrun_pid, poll_seconds)
    summary = convergence_summary(run)
    if summary.converged_block is not None:
        logger.info("converged at block %d (%s ns)", summary.converged_block, summary.converged_ns)
        stop_mdrun(mdrun_pid)

    u.load_new(str(xtc))
    trajectory_end_ns = frame_time_ns(u, n_complete_frames(u, mdrun_pid) - 1)
    options = {"system_options": system_options, "hydration_options": hydration_options,
               "block_options": block_options, "mdrun_pid": mdrun_pid, "poll_seconds": poll_seconds,
               "fp_round_decimals": fp_round_decimals, "anchor_confidence": anchor_confidence}
    metadata = provenance({"tpr": tpr, "xtc": xtc}) | {
        "ligand_selection": sel.ligand_selection, "water_selection": sel.water_selection, "options": options}
    run = run.assign_attrs(trajectory_end_ns=trajectory_end_ns)
    write_run(outdir, run, metadata)
    write_charts(run, outdir)
    if summary.converged_block is None:
        logger.info("DONE, not converged: the trajectory ended at %s ns, after %d complete block(s)",
                    trajectory_end_ns, summary.n_blocks)
        return

    estimate = estimate_at(run, summary.converged_block)
    try:
        write_selection(ligand, estimate, outdir / "selection", fp_round_decimals)
    except InputError as e:
        logger.warning("no G1/G2 selection: %s", e)
    write_selection_draws(ligand, estimate, outdir / "selection", fp_round_decimals)
