"""
Block-wise hydration RDF/fingerprint (FP) convergence monitor for a
ligand-in-water unbiased MD run.

This script owns only the ONLINE, BLOCK-WISE, CONVERGENCE-TESTING logic
(splitting an accumulating trajectory into blocks, comparing consecutive
blocks, deciding when to stop). The actual RDF and WaterFP math is not
duplicated here - it is imported from the sibling `waterfp` subpackage,
which implements those calculations independently of any
convergence/blocking concept:

  - waterfp.fingerprint.compute_density_profile()    - per-atom water density n(r)
  - waterfp.fingerprint.fp_from_density_profile()    - n(r) -> FP, g(r)

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
"""
import argparse
import os
import json
import time
import signal

import numpy as np
import pandas as pd
import MDAnalysis as mda
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ligand_waterfp.selections import select_heavy_atoms, select_water_oxygens
from ..waterfp.fingerprint import (
    NORM_TAIL_BINS_DEFAULT,
    compute_density_profile,
    fp_from_density_profile,
    make_bins,
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
    p.add_argument("--water-resname", default=None,
                    help="Residue name of the solvent (default: auto-detect via MDAnalysis's "
                         "'water' selection keyword, which knows the standard water resnames "
                         "across force fields). Only needed for nonstandard water naming.")
    p.add_argument("--water-atom-name", default=None,
                    help="Atom name of the water oxygen (default: atoms named O* within the "
                         "water residues). Only needed for nonstandard water naming.")
    p.add_argument("--rmax-nm", type=float, default=2.001)
    p.add_argument("--binwidth-nm", type=float, default=0.001)
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


def nrmsd(g_a, g_b):
    rmsd = np.sqrt(np.mean((g_a - g_b) ** 2))
    norm = max(np.max(g_a), np.max(g_b), 1e-8)
    return rmsd / norm


def main():
    args = parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "rdf_plots"), exist_ok=True)
    os.makedirs(os.path.join(args.outdir, "fp_plots"), exist_ok=True)

    bins = make_bins(args.rmax_nm, args.binwidth_nm)
    centers_nm = bins.centers_nm

    def get_universe():
        return mda.Universe(args.tpr, args.xtc)

    u0 = get_universe()
    lig_heavy = select_heavy_atoms(u0, args.ligand_resname, args.tpr)
    heavy_names = list(lig_heavy.names)
    heavy_indices = lig_heavy.indices
    water_O = select_water_oxygens(u0, args.tpr, args.water_resname, args.water_atom_name)
    water_O_idx = water_O.indices
    n_heavy = len(heavy_names)
    print(f"[monitor] {n_heavy} ligand heavy atoms (resname={args.ligand_resname}): {heavy_names}")
    print(f"[monitor] {len(water_O_idx)} water oxygens "
          f"(resname={args.water_resname or 'auto (water keyword)'}, "
          f"atom name={args.water_atom_name or 'O*'})")
    print(f"[monitor] RDF: rmax={args.rmax_nm} nm, bins={len(centers_nm)}, "
          f"norm=mean of last {args.norm_tail_bins} bins (WaterFP fp.py convention)")

    ts_per_frame_ps = None
    fp_rows = []
    rdf_rows = []
    metric_rows = []

    prev_fp = None
    prev_g = None
    prev_rank = None
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
        t_start_ns = u.trajectory[start_f].time / 1000.0
        t_end_ns = u.trajectory[end_f - 1].time / 1000.0

        n_r = compute_density_profile(
            u.atoms[heavy_indices],
            u.atoms[water_O_idx],
            start_f,
            end_f,
            bins,
        )

        fp_this_block = {}
        g_this_block = {}
        for ai, name in enumerate(heavy_names):
            fp_val, g_val, _norm = fp_from_density_profile(n_r[ai], centers_nm, args.norm_tail_bins)
            fp_this_block[name] = float(fp_val)
            g_this_block[name] = g_val

        for name in heavy_names:
            fp_rows.append(
                {"block": block_idx, "t_start_ns": t_start_ns, "t_end_ns": t_end_ns,
                 "t_mid_ns": 0.5 * (t_start_ns + t_end_ns), "atom": name, "FP": fp_this_block[name]}
            )
            for rbin, gval in zip(centers_nm, g_this_block[name]):
                rdf_rows.append({"block": block_idx, "atom": name, "r_nm": rbin, "g_r": gval})

        rank_now = pd.Series(fp_this_block).rank(ascending=False)

        if prev_fp is not None:
            rel_changes = {
                name: abs(fp_this_block[name] - prev_fp[name]) / max(abs(prev_fp[name]), 1e-6)
                for name in heavy_names
            }
            max_rel_change = max(rel_changes.values())
            nrmsds = {name: nrmsd(g_this_block[name], prev_g[name]) for name in heavy_names}
            max_nrmsd = max(nrmsds.values())
            rho, _ = spearmanr(rank_now.values, prev_rank.values)

            value_ok = max_rel_change <= args.fp_rel_tol
            rank_ok = rho >= args.spearman_tol
            rdf_ok = max_nrmsd <= args.rdf_nrmsd_tol
            block_stable = value_ok and rank_ok and rdf_ok
            stable_streak = stable_streak + 1 if block_stable else 0

            metric_rows.append({
                "block_transition": f"{block_idx-1}->{block_idx}",
                "t_end_ns": t_end_ns,
                "max_FP_relative_change": max_rel_change,
                "max_RDF_nRMSD": max_nrmsd,
                "spearman_rank_corr": rho,
                "value_stable": value_ok,
                "rdf_stable": rdf_ok,
                "rank_stable": rank_ok,
                "block_stable": block_stable,
                "stable_streak": stable_streak,
            })
            print(f"[monitor] block {block_idx} (t={t_end_ns:.1f} ns): "
                  f"max|dFP/FP|={max_rel_change:.3f} max_nRMSD_RDF={max_nrmsd:.3f} "
                  f"spearman={rho:.3f} stable={block_stable} streak={stable_streak}")

            if stable_streak >= args.n_stable:
                converged = True
                stop_reason = (
                    f"RDF profile, FP values, and FP-based heavy-atom ranking were all stable "
                    f"(max FP relative change <= {args.fp_rel_tol*100:.0f}%, max RDF normalised RMSD "
                    f"<= {args.rdf_nrmsd_tol*100:.0f}%, Spearman rank correlation >= {args.spearman_tol}) "
                    f"across {args.n_stable} consecutive block-to-block transitions."
                )
                stop_time_ns = t_end_ns
                stop_block = block_idx

        pd.DataFrame(fp_rows).to_csv(os.path.join(args.outdir, "fp_values_per_block.csv"), index=False)
        pd.DataFrame(rdf_rows).to_csv(os.path.join(args.outdir, "rdf_profiles_per_block.csv"), index=False)
        pd.DataFrame(metric_rows).to_csv(os.path.join(args.outdir, "convergence_metrics.csv"), index=False)

        prev_fp = fp_this_block
        prev_g = g_this_block
        prev_rank = rank_now
        block_idx += 1

        if converged:
            break

    fp_df = pd.DataFrame(fp_rows)
    rdf_df = pd.DataFrame(rdf_rows)

    for name in heavy_names:
        sub = fp_df[fp_df["atom"] == name].sort_values("t_mid_ns")
        fig, ax = plt.subplots(figsize=(5, 3.5))
        ax.plot(sub["t_mid_ns"], sub["FP"], marker="o", color="tab:blue")
        ax.set_xlabel("simulation time (ns)")
        ax.set_ylabel("FP (WaterFP excess-entropy integral)")
        ax.set_title(f"Hydration FP vs time - {name}")
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "fp_plots", f"FP_vs_time_{name}.png"), dpi=150)
        plt.close(fig)

        fig2, ax2 = plt.subplots(figsize=(5, 3.5))
        rsub = rdf_df[rdf_df["atom"] == name]
        for b in sorted(rsub["block"].unique()):
            bb = rsub[rsub["block"] == b]
            ax2.plot(bb["r_nm"], bb["g_r"], alpha=0.6, label=f"block {b}")
        ax2.set_xlabel("r (nm) to water O")
        ax2.set_ylabel("g(r)")
        ax2.set_xlim(0, 1.0)
        ax2.set_title(f"RDF per block - {name}")
        if rsub["block"].nunique() <= 8:
            ax2.legend(fontsize=6)
        fig2.tight_layout()
        fig2.savefig(os.path.join(args.outdir, "rdf_plots", f"RDF_per_block_{name}.png"), dpi=150)
        plt.close(fig2)

    rank_table = fp_df.pivot(index="atom", columns="block", values="FP")
    rank_table_ranked = rank_table.rank(ascending=False, axis=0)
    rank_table.to_csv(os.path.join(args.outdir, "fp_values_by_block_wide.csv"))
    rank_table_ranked.to_csv(os.path.join(args.outdir, "fp_ranking_by_block_wide.csv"))

    summary = {
        "converged": converged,
        "stop_reason": stop_reason,
        "stop_block": stop_block,
        "stop_time_ns": stop_time_ns,
        "block_size_ns": args.block_ns,
        "n_stable_transitions_required": args.n_stable,
        "thresholds": {
            "FP_relative_change_tol": args.fp_rel_tol,
            "RDF_normalised_RMSD_tol": args.rdf_nrmsd_tol,
            "spearman_rank_corr_tol": args.spearman_tol,
        },
        "FP_method": "WaterFP (github.com/valeriorizzi/WaterFP, Scripts/fp.py): "
                     "FP = trapz(-2*pi*norm*(g*ln(g)-g+1)*r^2, dx) over r in [0,rmax] nm, "
                     "g(r)=n(r)/norm, norm=mean of the last norm_tail_bins bins of the raw "
                     "water-O number-density profile n(r) around the ligand heavy atom.",
        "ligand_resname": args.ligand_resname,
        "ligand_heavy_atoms": heavy_names,
        "n_blocks_completed": block_idx,
    }
    with open(os.path.join(args.outdir, "convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(args.outdir, "convergence_summary.txt"), "w") as f:
        f.write("WaterFP-style hydration RDF/FP block-wise convergence monitor summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Stop block: {stop_block}\n")
        f.write(f"Stop simulation time: {stop_time_ns} ns\n")
        f.write(f"Reason: {stop_reason}\n")
        f.write(f"Block size: {args.block_ns} ns\n")
        f.write(f"Consecutive stable transitions required: {args.n_stable}\n")
        f.write(f"Thresholds: FP rel. change <= {args.fp_rel_tol}, RDF nRMSD <= {args.rdf_nrmsd_tol}, "
                f"Spearman rho >= {args.spearman_tol}\n")
        f.write(f"FP method: {summary['FP_method']}\n")
        f.write(f"Ligand heavy atoms monitored ({len(heavy_names)}): {', '.join(heavy_names)}\n")

    print(f"[monitor] DONE. converged={converged} stop_block={stop_block} stop_time_ns={stop_time_ns}")
    print(f"[monitor] reason: {stop_reason}")

    if converged and process_alive(args.mdrun_pid):
        print(f"[monitor] convergence reached - sending SIGTERM to mdrun pid {args.mdrun_pid}")
        try:
            os.kill(args.mdrun_pid, signal.SIGTERM)
        except OSError as e:
            print(f"[monitor] failed to signal mdrun pid {args.mdrun_pid}: {e}")


if __name__ == "__main__":
    main()
