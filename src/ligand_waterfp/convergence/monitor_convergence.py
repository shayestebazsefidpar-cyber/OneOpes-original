"""
Convergence monitor for an unbiased ligand-in-water MD run.

Analyses the trajectory block by block as mdrun writes it (waterfp.WaterFP),
compares each block with the previous one (the convergence module), and stops
once --n-stable transitions in a row are stable, sending SIGTERM to the given
mdrun PID. Tables, plots and a summary are written to --outdir; on convergence,
the G1/G2 selection of the converged block goes to --outdir/selection.
METHOD_RATIONALE.md explains the criteria.

Usage:
    ligand-waterfp-monitor                       # no auto-stop, just monitor/report
    ligand-waterfp-monitor 12345                 # SIGTERM pid 12345 on convergence
    ligand-waterfp-monitor --tpr prod.tpr --xtc prod.xtc \\
        --ligand-resname MOL --water-resname SOL --water-atom-name O
"""
import argparse
import os
import time
import signal
from collections.abc import Callable, Iterator
from itertools import count
from pathlib import Path

import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence import (
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
from ligand_waterfp.g1_g2_selection.select_g1_g2 import write_selection
from trajectory_blocks import frame_step_ps, frame_time_ns, frames_per_block, time_span_ns
from waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("mdrun_pid", type=int, nargs="?", default=None,
                   help="PID of a running gmx mdrun to SIGTERM on convergence (optional)")
    p.add_argument("--tpr", default="prod.tpr")
    p.add_argument("--xtc", default="prod.xtc")
    p.add_argument("--outdir", default="outputs/convergence")
    p.add_argument("--ligand-resname", default="MOL",
                    help="Residue name of the ligand in the topology (default: MOL)")
    p.add_argument("--water-resname", default="SOL",
                    help="Residue name of the solvent (default: SOL)")
    p.add_argument("--water-atom-name", default="O",
                    help="Atom name of the water oxygen within --water-resname (default: O). "
                         "Water naming is NOT guaranteed consistent across force fields/files - "
                         "e.g. some topologies name the same water residue WAT/HOH with oxygen "
                         "atom name OW instead of SOL/O. Check your own .tpr before trusting the "
                         "defaults on a new system.")
    p.add_argument("--rmax-nm", type=float, default=RadialGrid.rmax_nm)
    p.add_argument("--binwidth-nm", type=float, default=RadialGrid.binwidth_nm)
    p.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    p.add_argument("--block-ns", type=float, default=5.0,
                    help="Trajectory block size in ns (default: 5.0)")
    p.add_argument("--fp-rel-tol", type=float, default=0.10,
                    help="Max allowed relative FP change between blocks (default: 0.10 = 10%%)")
    p.add_argument("--rdf-nrmsd-tol", type=float, default=0.15,
                    help="Max allowed normalised RMSD between consecutive g(r) curves (default: 0.15 = 15%%)")
    p.add_argument("--spearman-tol", type=float, default=0.90,
                    help="Min Spearman rank correlation between consecutive FP rankings (default: 0.90)")
    p.add_argument("--n-stable", type=int, default=3,
                    help="Consecutive stable block-transitions required to declare convergence (default: 3)")
    p.add_argument("--poll-seconds", type=int, default=30,
                    help="Wait time between polls when waiting for more trajectory frames (default: 30)")
    p.add_argument("--fp-round-decimals", type=int, default=1,
                    help="Decimals FP is rounded to when the G1/G2 selection compares atoms (default: 1)")
    return p.parse_args()


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


def main():
    args = parse_args()
    tol = Tolerances(
        fp_rel_tol=args.fp_rel_tol,
        rdf_nrmsd_tol=args.rdf_nrmsd_tol,
        spearman_tol=args.spearman_tol,
        n_stable=args.n_stable,
    )

    # Validate the required trajectory files before any expensive
    # processing, and echo the resolved paths this run is using.
    tpr_path = Path(args.tpr)
    xtc_path = Path(args.xtc)
    outdir = Path(args.outdir)

    if not tpr_path.is_file():
        raise SystemExit(f"[monitor] --tpr file not found: {tpr_path.resolve()}")
    if not xtc_path.is_file():
        raise SystemExit(f"[monitor] --xtc file not found: {xtc_path.resolve()}")

    print(f"[monitor] tpr: {tpr_path.resolve()}")
    print(f"[monitor] xtc: {xtc_path.resolve()}")
    print(f"[monitor] outdir: {outdir.resolve()}")

    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "rdf_plots").mkdir(parents=True, exist_ok=True)
    (outdir / "fp_plots").mkdir(parents=True, exist_ok=True)

    def get_universe():
        return mda.Universe(str(tpr_path), str(xtc_path))

    u0 = get_universe()
    lig_heavy = u0.select_atoms(f"resname {args.ligand_resname} and not name H*")
    if len(lig_heavy) == 0:
        raise SystemExit(
            f"No heavy atoms found for resname '{args.ligand_resname}' in {args.tpr}. "
            "Check --ligand-resname against your own topology."
        )
    heavy_names = list(lig_heavy.names)
    heavy_indices = lig_heavy.indices
    water_O = u0.select_atoms(f"resname {args.water_resname} and name {args.water_atom_name}")
    if len(water_O) == 0:
        raise SystemExit(
            f"No water oxygens found for resname '{args.water_resname}' / atom name "
            f"'{args.water_atom_name}' in {args.tpr}. Water naming is not standardized across "
            "force fields/files - check --water-resname/--water-atom-name against your own topology."
        )
    water_O_idx = water_O.indices
    n_heavy = len(heavy_names)
    print(f"[monitor] {n_heavy} ligand heavy atoms (resname={args.ligand_resname}): {heavy_names}")
    print(f"[monitor] {len(water_O_idx)} water oxygens "
          f"(resname={args.water_resname}, atom name={args.water_atom_name})")
    grid = RadialGrid(rmax_nm=args.rmax_nm, binwidth_nm=args.binwidth_nm)
    print(f"[monitor] RDF: rmax={grid.rmax_nm} nm, bins={grid.n_bins}, "
          f"norm=mean of last {args.norm_tail_bins} bins (WaterFP fp.py convention)")

    u = wait_for_frames(get_universe, 2, args.mdrun_pid, args.poll_seconds)
    try:
        dt_ps = frame_step_ps(u)
        block_length = frames_per_block(args.block_ns, dt_ps)
    except ValueError as e:
        raise SystemExit(f"[monitor] {e}")
    print(f"[monitor] dt/frame = {dt_ps} ps -> {block_length} frames/block ({args.block_ns} ns)")

    blocks, block_times, stabilities = [], [], []
    for u, frames in completed_blocks(get_universe, block_length, args.mdrun_pid, args.poll_seconds):
        blocks.append(WaterFP(u.atoms[heavy_indices], u.atoms[water_O_idx], grid=grid,
                              norm_tail_bins=args.norm_tail_bins).run(start=frames.start, stop=frames.stop))
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
    summary = summarize(stabilities, block_times, args.block_ns, tol)
    trajectory_end_ns = frame_time_ns(get_universe(), -1)
    write_summary(outdir / "convergence_summary.json", summary, trajectory_end_ns=trajectory_end_ns,
                  ligand_resname=args.ligand_resname, ligand_heavy_atoms=heavy_names)

    fp_df = fp_per_block(blocks, block_times)
    rdf_df = rdf_per_block(blocks)
    for name in heavy_names:
        plot_atom(name, fp_df, rdf_df, outdir)

    if not summary.converged:
        print(f"[monitor] DONE, not converged: the trajectory ended at {trajectory_end_ns} ns, "
              f"after {summary.n_blocks} complete block(s)")
        return
    print(f"[monitor] DONE, converged at block {summary.converged_block} ({summary.converged_ns} ns)")

    if process_alive(args.mdrun_pid):
        print(f"[monitor] sending SIGTERM to mdrun pid {args.mdrun_pid}")
        try:
            os.kill(args.mdrun_pid, signal.SIGTERM)
        except OSError as e:
            print(f"[monitor] failed to signal mdrun pid {args.mdrun_pid}: {e}")

    try:
        selection = write_selection(blocks[-1], outdir / "selection", args.fp_round_decimals)
        print(f"[monitor] {selection}, written to {outdir / 'selection'}")
    except ValueError as e:
        print(f"[monitor] no G1/G2 selection for block {summary.converged_block}: {e}")
