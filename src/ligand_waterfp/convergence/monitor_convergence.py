"""
Block-wise hydration RDF/fingerprint (FP) convergence monitor for a
ligand-in-water unbiased MD run.

This script owns only the ONLINE, BLOCK-WISE, CONVERGENCE-TESTING logic
(splitting an accumulating trajectory into blocks, comparing consecutive
blocks, deciding when to stop). The actual RDF and WaterFP math is not
duplicated here - each block is analysed with the shared `waterfp.WaterFP`
analysis (FP and g(r) per atom), which has no notion of convergence or
blocks.

For each block of --block-ns newly-accumulated nanoseconds, this script
tracks, per ligand heavy atom:
  (a) RDF-profile stability: normalised RMSD of g(r) vs. the previous block
  (b) FP stability: relative change of the scalar FP vs. the previous block
  (c) Ranking stability: Spearman correlation of the FP-based atom ranking
      vs. the previous block

Convergence = ALL THREE criteria hold for --n-stable consecutive
block-to-block transitions. On convergence, and if a running mdrun PID was
given, SIGTERM is sent to stop production early. All numbers/plots/summary
are written under the output directory. See METHOD_RATIONALE.md in this
folder for the full reasoning behind the compound criterion and the
stopping rule.

Does NOT perform G1/G2 atom selection or any downstream CV construction -
see the official_selection subpackage onward for that.

Usage (after `pip install -e .` from the package root):
    ligand-waterfp-monitor [mdrun_pid] [options]
    python -m ligand_waterfp.convergence.monitor_convergence [mdrun_pid] [options]

    ligand-waterfp-monitor                       # no auto-stop, just monitor/report
    ligand-waterfp-monitor 12345                  # SIGTERM pid 12345 on convergence
    ligand-waterfp-monitor --tpr prod.tpr --xtc prod.xtc \\
        --ligand-resname MOL --water-resname SOL --water-atom-name O

--- Module structure (refactored per code-review Rules 2 and 5) ---
The computational/reporting pieces main() used to do inline are pulled out
into their own functions, each returning data rather than printing it:

    plot_atom()        - one atom's accumulated history -> two PNG files
    write_reports()    - the full run's accumulated history -> summary
                          dict + the wide-format CSVs/JSON/TXT report files

Each block itself is analysed directly with `waterfp.WaterFP`, and
consecutive blocks are compared with the shared `convergence.check_stability`.

main() itself only coordinates: it owns the block-polling loop (which
needs live state - a growing trajectory, streak counters, accumulator
lists - that doesn't belong in a pure function) and is the only place
that prints progress or writes the human-facing summary text, per Rule 5.
No formula, threshold, output filename, CSV/JSON schema, or printed
message was changed by this refactor - see METHOD_RATIONALE.md for the
science, which this change does not touch.

--- Structured values ---
check_stability() returns a StabilityResult; Tolerances bundles the 4
threshold/streak values main() builds once from the CLI args (both from
the `convergence` module). Report field names are mapped explicitly from
these dataclasses.
"""
import argparse
import os
import json
import time
import signal
from pathlib import Path

import pandas as pd
import MDAnalysis as mda
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from convergence import Tolerances, check_stability
from waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP

FP_METHOD_TEXT = (
    "WaterFP (github.com/valeriorizzi/WaterFP, Scripts/fp.py): "
    "FP = trapz(-2*pi*norm*(g*ln(g)-g+1)*r^2, dx) over r in [0,rmax] nm, "
    "g(r)=n(r)/norm, norm=mean of the last norm_tail_bins bins of the raw "
    "water-O number-density profile n(r) around the ligand heavy atom."
)


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
    return p.parse_args()


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def write_block_rows(table: pd.DataFrame, path: Path, first_block: bool) -> None:
    """Write one block's rows to ``path``: a new file (with header) for the first block, appended after."""
    table.to_csv(path, mode="w" if first_block else "a", header=first_block, index=False)


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


def write_reports(outdir, fp_df, converged, stop_reason, stop_block, stop_time_ns,
                   block_ns, tol: Tolerances, ligand_resname, heavy_names, n_blocks_completed):
    """Write the end-of-run report files, exactly as main() used to build
    them inline: the wide-format FP/ranking-by-block CSVs (pivoted from
    fp_df), convergence_summary.json, and convergence_summary.txt. Same
    schema, same field names, same text - only extracted out of main().
    No printing (Rule 5) - returns the summary dict main() prints from.

    tol supplies the threshold/streak values written into the JSON/TXT
    summary below.

    Does not write the per-block incremental CSVs (fp_values_per_block.csv,
    rdf_profiles_per_block.csv, convergence_metrics.csv) - those are
    written after every block, inside the loop, for resumability, which is
    a different concern from this end-of-run reporting step; main() still
    writes them directly (see the loop body), same as before.
    """
    outdir = Path(outdir)
    rank_table = fp_df.pivot(index="atom", columns="block", values="FP")
    rank_table_ranked = rank_table.rank(ascending=False, axis=0)
    rank_table.to_csv(outdir / "fp_values_by_block_wide.csv")
    rank_table_ranked.to_csv(outdir / "fp_ranking_by_block_wide.csv")

    summary = {
        "converged": converged,
        "stop_reason": stop_reason,
        "stop_block": stop_block,
        "stop_time_ns": stop_time_ns,
        "block_size_ns": block_ns,
        "n_stable_transitions_required": tol.n_stable,
        "thresholds": {
            "FP_relative_change_tol": tol.fp_rel_tol,
            "RDF_normalised_RMSD_tol": tol.rdf_nrmsd_tol,
            "spearman_rank_corr_tol": tol.spearman_tol,
        },
        "FP_method": FP_METHOD_TEXT,
        "ligand_resname": ligand_resname,
        "ligand_heavy_atoms": heavy_names,
        "n_blocks_completed": n_blocks_completed,
    }
    with open(outdir / "convergence_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    with open(outdir / "convergence_summary.txt", "w") as f:
        f.write("WaterFP-style hydration RDF/FP block-wise convergence monitor summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Stop block: {stop_block}\n")
        f.write(f"Stop simulation time: {stop_time_ns} ns\n")
        f.write(f"Reason: {stop_reason}\n")
        f.write(f"Block size: {block_ns} ns\n")
        f.write(f"Consecutive stable transitions required: {tol.n_stable}\n")
        f.write(f"Thresholds: FP rel. change <= {tol.fp_rel_tol}, RDF nRMSD <= {tol.rdf_nrmsd_tol}, "
                f"Spearman rho >= {tol.spearman_tol}\n")
        f.write(f"FP method: {summary['FP_method']}\n")
        f.write(f"Ligand heavy atoms monitored ({len(heavy_names)}): {', '.join(heavy_names)}\n")

    return summary


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

    ts_per_frame_ps = None
    fp_tables = []   # one table per block: fp_values_per_block.csv rows
    rdf_tables = []  # one table per block: rdf_profiles_per_block.csv rows
    metric_rows = []

    previous = None  # the previous block's WaterFP analysis
    stable_streak = 0
    block_idx = 0
    converged = False
    stop_reason = None
    stop_time_ns = None
    stop_block = None
    frames_per_block = None

    while True:
        u = get_universe()
        if ts_per_frame_ps is None and len(u.trajectory) >= 2:
            ts_per_frame_ps = u.trajectory[1].time - u.trajectory[0].time
            frames_per_block = int(round(args.block_ns * 1000.0 / ts_per_frame_ps))
            print(f"[monitor] dt/frame = {ts_per_frame_ps} ps -> {frames_per_block} frames/block ({args.block_ns} ns)")

        n_frames = len(u.trajectory)
        if frames_per_block is None:
            time.sleep(args.poll_seconds)
            continue

        needed = (block_idx + 1) * frames_per_block
        if n_frames < needed:
            if not process_alive(args.mdrun_pid):
                stop_reason = (
                    f"mdrun process (pid {args.mdrun_pid}) is no longer running before block "
                    f"{block_idx} could complete ({n_frames} frames available, {needed} needed). "
                    "Simulation ended (finished its full length, crashed, or was stopped externally) "
                    "before convergence was detected by this monitor."
                )
                stop_time_ns = u.trajectory[-1].time / 1000.0 if n_frames > 0 else 0.0
                stop_block = block_idx
                break
            time.sleep(args.poll_seconds)
            continue

        start_f = block_idx * frames_per_block
        end_f = needed

        block = WaterFP(u.atoms[heavy_indices], u.atoms[water_O_idx], grid=grid,
                        norm_tail_bins=args.norm_tail_bins)
        block.run(start=start_f, stop=end_f)
        t_start_ns, t_end_ns = block.times[0] / 1000.0, block.times[-1] / 1000.0

        fp_table = block.fp_table().assign(block=block_idx, t_start_ns=t_start_ns, t_end_ns=t_end_ns,
                                           t_mid_ns=0.5 * (t_start_ns + t_end_ns))
        fp_tables.append(fp_table[["block", "t_start_ns", "t_end_ns", "t_mid_ns", "atom", "FP"]])
        rdf_tables.append(block.profile_table().assign(block=block_idx)[["block", "atom", "r_nm", "g_r"]])

        if previous is not None:
            stability = check_stability(block.results, previous.results, tol)
            block_stable = stability.passed
            stable_streak = stable_streak + 1 if block_stable else 0

            metric_rows.append({
                "block_transition": f"{block_idx-1}->{block_idx}",
                "t_end_ns": t_end_ns,
                "max_FP_relative_change": stability.fp_change,
                "max_RDF_nRMSD": stability.rdf_nrmsd,
                "spearman_rank_corr": stability.spearman_rho,
                "value_stable": stability.fp_pass,
                "rdf_stable": stability.rdf_pass,
                "rank_stable": stability.spearman_pass,
                "block_stable": block_stable,
                "stable_streak": stable_streak,
            })
            print(f"[monitor] block {block_idx} (t={t_end_ns:.1f} ns): "
                  f"max|dFP/FP|={stability.fp_change:.3f} "
                  f"max_nRMSD_RDF={stability.rdf_nrmsd:.3f} "
                  f"spearman={stability.spearman_rho:.3f} stable={block_stable} streak={stable_streak}")

            if stable_streak >= tol.n_stable:
                converged = True
                stop_reason = (
                    f"RDF profile, FP values, and FP-based heavy-atom ranking were all stable "
                    f"(max FP relative change <= {tol.fp_rel_tol*100:.0f}%, max RDF normalised RMSD "
                    f"<= {tol.rdf_nrmsd_tol*100:.0f}%, Spearman rank correlation >= {tol.spearman_tol}) "
                    f"across {tol.n_stable} consecutive block-to-block transitions."
                )
                stop_time_ns = t_end_ns
                stop_block = block_idx

        write_block_rows(fp_tables[-1], outdir / "fp_values_per_block.csv", first_block=block_idx == 0)
        write_block_rows(rdf_tables[-1], outdir / "rdf_profiles_per_block.csv", first_block=block_idx == 0)
        pd.DataFrame(metric_rows).to_csv(outdir / "convergence_metrics.csv", index=False)

        previous = block
        block_idx += 1

        if converged:
            break

    fp_df = pd.concat(fp_tables, ignore_index=True)
    rdf_df = pd.concat(rdf_tables, ignore_index=True)

    for name in heavy_names:
        plot_atom(name, fp_df, rdf_df, outdir)

    summary = write_reports(
        outdir, fp_df, converged, stop_reason, stop_block, stop_time_ns,
        args.block_ns, tol, args.ligand_resname, heavy_names, block_idx,
    )

    print(f"[monitor] DONE. converged={summary['converged']} stop_block={summary['stop_block']} "
          f"stop_time_ns={summary['stop_time_ns']}")
    print(f"[monitor] reason: {summary['stop_reason']}")

    if converged and process_alive(args.mdrun_pid):
        print(f"[monitor] convergence reached - sending SIGTERM to mdrun pid {args.mdrun_pid}")
        try:
            os.kill(args.mdrun_pid, signal.SIGTERM)
        except OSError as e:
            print(f"[monitor] failed to signal mdrun pid {args.mdrun_pid}: {e}")
