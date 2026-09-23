"""
Block-wise hydration RDF/fingerprint (FP) convergence monitor for the
ligand-in-water unbiased MD run in this directory (prod.tpr/prod.xtc).

RDF and FP are computed exactly as in the reference WaterFP repository
(https://github.com/valeriorizzi/WaterFP/tree/main/Scripts, calc_rdf.sh +
fp.py), reimplemented on the fly per accumulated trajectory block instead of
gmx rdf on a single fixed trajectory:

  - per ligand heavy atom, radial number-density histogram n(r) of water
    oxygens (OW) out to r_max=2.001 nm, bin width 0.001 nm (2001 bins)
  - "norm" = mean of the last 500 bins of n(r) (empirical bulk-density
    plateau, exactly as fp.py does - no separate bulk density is assumed)
  - dimensionless g(r) = n(r) / norm
  - FP = trapz( -2*pi*norm*(g*ln(g) - g + 1)*r^2 , dx=0.001 nm ) over the
    full r range (the g->0 limit of the bracket, which is 1, is substituted
    where g==0, exactly as in fp.py)

For each block of BLOCK_NS newly-accumulated nanoseconds, this script tracks,
per ligand heavy atom:
  (a) RDF-profile stability: normalised RMSD of g(r) vs. the previous block
  (b) FP stability: relative change of the scalar FP vs. the previous block
  (c) Ranking stability: Spearman correlation of the FP-based atom ranking
      vs. the previous block

Convergence = ALL THREE criteria hold for N_STABLE_NEEDED consecutive
block-to-block transitions. On convergence, the production mdrun process is
sent SIGTERM to stop the MD early. All numbers/metrics/plots/summary are
written under ./convergence/.

Does NOT perform POI selection, OPES, or any downstream analysis.
"""
import os
import sys
import json
import time
import signal
import numpy as np
import pandas as pd
import MDAnalysis as mda
from MDAnalysis.lib.distances import distance_array
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_trapz = getattr(np, "trapezoid", None) or np.trapz

TPR = "prod.tpr"
XTC = "prod.xtc"
OUTDIR = "convergence"
MDRUN_PID = int(sys.argv[1]) if len(sys.argv) > 1 else None

BLOCK_NS = 5.0

# --- RDF/FP settings, matching WaterFP's calc_rdf.sh / fp.py exactly ---
RDF_RMAX_NM = 2.001
RDF_BINWIDTH_NM = 0.001
RDF_NBINS = int(round(RDF_RMAX_NM / RDF_BINWIDTH_NM))   # 2001
NORM_TAIL_BINS = 500

# --- convergence thresholds (not specified by WaterFP; chosen here and
#     fully documented in the saved summary) ---
VALUE_REL_TOL = 0.10        # 10% relative change in FP
RANK_SPEARMAN_TOL = 0.90
RDF_NRMSD_TOL = 0.15        # 15% normalised RMSD between consecutive g(r)
N_STABLE_NEEDED = 3
POLL_SECONDS = 30

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "rdf_plots"), exist_ok=True)
os.makedirs(os.path.join(OUTDIR, "fp_plots"), exist_ok=True)

RDF_EDGES_NM = np.linspace(0.0, RDF_RMAX_NM, RDF_NBINS + 1)
RDF_CENTERS_NM = 0.5 * (RDF_EDGES_NM[:-1] + RDF_EDGES_NM[1:])
RDF_EDGES_A = RDF_EDGES_NM * 10.0   # MDAnalysis distances are in Angstrom
SHELL_VOL_NM3 = (4.0 / 3.0) * np.pi * (RDF_EDGES_NM[1:] ** 3 - RDF_EDGES_NM[:-1] ** 3)


def process_alive(pid):
    if pid is None:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def get_universe():
    return mda.Universe(TPR, XTC)


def fp_from_density_profile(n_r, norm):
    """Reimplementation of fp.py's integrand, given the raw number-density
    profile n_r (waters/nm^3 per bin) and its own bulk-tail 'norm'."""
    g = n_r / norm
    r = RDF_CENTERS_NM
    integrand = np.empty_like(g)
    mask0 = (g == 0)
    integrand[mask0] = -2 * np.pi * norm * (r[mask0] ** 2)  # lim g->0 of (g ln g - g + 1) = 1
    gnz = g[~mask0]
    integrand[~mask0] = -2 * np.pi * norm * (gnz * np.log(gnz) - gnz + 1) * (r[~mask0] ** 2)
    fp = _trapz(integrand, dx=RDF_BINWIDTH_NM)
    return fp, g


def block_density_profiles(u, heavy_indices, water_O_idx, start_f, end_f):
    """Per-atom raw number-density histogram n(r) (waters/nm^3), averaged
    over all frames in [start_f, end_f)."""
    n_heavy = len(heavy_indices)
    hist_sum = np.zeros((n_heavy, RDF_NBINS))
    n_frames = end_f - start_f
    for fi in range(start_f, end_f):
        u.trajectory[fi]
        heavy_pos = u.atoms.positions[heavy_indices]
        water_pos = u.atoms.positions[water_O_idx]
        d = distance_array(heavy_pos, water_pos, box=u.dimensions)  # Angstrom
        for ai in range(n_heavy):
            h, _ = np.histogram(d[ai], bins=RDF_EDGES_A)
            hist_sum[ai] += h
    mean_hist = hist_sum / n_frames         # mean count per bin per frame
    n_r = mean_hist / SHELL_VOL_NM3         # number density, waters/nm^3
    return n_r


def nrmsd(g_a, g_b):
    rmsd = np.sqrt(np.mean((g_a - g_b) ** 2))
    norm = max(np.max(g_a), np.max(g_b), 1e-8)
    return rmsd / norm


def main():
    u0 = get_universe()
    lig_heavy = u0.select_atoms("resname MOL and not name H*")
    heavy_names = list(lig_heavy.names)
    heavy_indices = lig_heavy.indices
    water_O = u0.select_atoms("resname SOL and name O")
    water_O_idx = water_O.indices
    n_heavy = len(heavy_names)
    print(f"[monitor] {n_heavy} ligand heavy atoms: {heavy_names}")
    print(f"[monitor] {len(water_O_idx)} water oxygens")
    print(f"[monitor] RDF: rmax={RDF_RMAX_NM} nm, bins={RDF_NBINS}, "
          f"norm=mean of last {NORM_TAIL_BINS} bins (WaterFP fp.py convention)")

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
            frames_per_block = int(round(BLOCK_NS * 1000.0 / ts_per_frame_ps))
            print(f"[monitor] dt/frame = {ts_per_frame_ps} ps -> {frames_per_block} frames/block ({BLOCK_NS} ns)")

        n_frames = len(u.trajectory)
        if frames_per_block is None:
            time.sleep(POLL_SECONDS)
            continue

        needed = (block_idx + 1) * frames_per_block
        if n_frames < needed:
            if not process_alive(MDRUN_PID):
                stop_reason = (
                    f"mdrun process (pid {MDRUN_PID}) is no longer running before block "
                    f"{block_idx} could complete ({n_frames} frames available, {needed} needed). "
                    "Simulation ended (finished its full length, crashed, or was stopped externally) "
                    "before convergence was detected by this monitor."
                )
                stop_time_ns = u.trajectory[-1].time / 1000.0 if n_frames > 0 else 0.0
                stop_block = block_idx
                break
            time.sleep(POLL_SECONDS)
            continue

        start_f = block_idx * frames_per_block
        end_f = needed
        t_start_ns = u.trajectory[start_f].time / 1000.0
        t_end_ns = u.trajectory[end_f - 1].time / 1000.0

        n_r = block_density_profiles(u, heavy_indices, water_O_idx, start_f, end_f)

        fp_this_block = {}
        g_this_block = {}
        for ai, name in enumerate(heavy_names):
            norm = np.mean(n_r[ai, -NORM_TAIL_BINS:])
            fp_val, g_val = fp_from_density_profile(n_r[ai], norm)
            fp_this_block[name] = float(fp_val)
            g_this_block[name] = g_val

        for name in heavy_names:
            fp_rows.append(
                {"block": block_idx, "t_start_ns": t_start_ns, "t_end_ns": t_end_ns,
                 "t_mid_ns": 0.5 * (t_start_ns + t_end_ns), "atom": name, "FP": fp_this_block[name]}
            )
            for rbin, gval in zip(RDF_CENTERS_NM, g_this_block[name]):
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

            value_ok = max_rel_change <= VALUE_REL_TOL
            rank_ok = rho >= RANK_SPEARMAN_TOL
            rdf_ok = max_nrmsd <= RDF_NRMSD_TOL
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

            if stable_streak >= N_STABLE_NEEDED:
                converged = True
                stop_reason = (
                    f"RDF profile, FP values, and FP-based heavy-atom ranking were all stable "
                    f"(max FP relative change <= {VALUE_REL_TOL*100:.0f}%, max RDF normalised RMSD "
                    f"<= {RDF_NRMSD_TOL*100:.0f}%, Spearman rank correlation >= {RANK_SPEARMAN_TOL}) "
                    f"across {N_STABLE_NEEDED} consecutive block-to-block transitions."
                )
                stop_time_ns = t_end_ns
                stop_block = block_idx

        pd.DataFrame(fp_rows).to_csv(os.path.join(OUTDIR, "fp_values_per_block.csv"), index=False)
        pd.DataFrame(rdf_rows).to_csv(os.path.join(OUTDIR, "rdf_profiles_per_block.csv"), index=False)
        pd.DataFrame(metric_rows).to_csv(os.path.join(OUTDIR, "convergence_metrics.csv"), index=False)

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
        fig.savefig(os.path.join(OUTDIR, "fp_plots", f"FP_vs_time_{name}.png"), dpi=150)
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
        fig2.savefig(os.path.join(OUTDIR, "rdf_plots", f"RDF_per_block_{name}.png"), dpi=150)
        plt.close(fig2)

    rank_table = fp_df.pivot(index="atom", columns="block", values="FP")
    rank_table_ranked = rank_table.rank(ascending=False, axis=0)
    rank_table.to_csv(os.path.join(OUTDIR, "fp_values_by_block_wide.csv"))
    rank_table_ranked.to_csv(os.path.join(OUTDIR, "fp_ranking_by_block_wide.csv"))

    summary = {
        "converged": converged,
        "stop_reason": stop_reason,
        "stop_block": stop_block,
        "stop_time_ns": stop_time_ns,
        "block_size_ns": BLOCK_NS,
        "n_stable_transitions_required": N_STABLE_NEEDED,
        "thresholds": {
            "FP_relative_change_tol": VALUE_REL_TOL,
            "RDF_normalised_RMSD_tol": RDF_NRMSD_TOL,
            "spearman_rank_corr_tol": RANK_SPEARMAN_TOL,
        },
        "FP_method": "WaterFP (github.com/valeriorizzi/WaterFP, Scripts/fp.py): "
                     "FP = trapz(-2*pi*norm*(g*ln(g)-g+1)*r^2, dx=0.001 nm) over r in [0,2.001] nm, "
                     "g(r)=n(r)/norm, norm=mean of last 500 of 2001 bins of the raw water-O "
                     "number-density profile n(r) around the ligand heavy atom.",
        "ligand_heavy_atoms": heavy_names,
        "n_blocks_completed": block_idx,
    }
    with open(os.path.join(OUTDIR, "convergence_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    with open(os.path.join(OUTDIR, "convergence_summary.txt"), "w") as f:
        f.write("WaterFP-style hydration RDF/FP block-wise convergence monitor summary\n")
        f.write("=" * 70 + "\n")
        f.write(f"Converged: {converged}\n")
        f.write(f"Stop block: {stop_block}\n")
        f.write(f"Stop simulation time: {stop_time_ns} ns\n")
        f.write(f"Reason: {stop_reason}\n")
        f.write(f"Block size: {BLOCK_NS} ns\n")
        f.write(f"Consecutive stable transitions required: {N_STABLE_NEEDED}\n")
        f.write(f"Thresholds: FP rel. change <= {VALUE_REL_TOL}, RDF nRMSD <= {RDF_NRMSD_TOL}, "
                f"Spearman rho >= {RANK_SPEARMAN_TOL}\n")
        f.write(f"FP method: {summary['FP_method']}\n")
        f.write(f"Ligand heavy atoms monitored ({len(heavy_names)}): {', '.join(heavy_names)}\n")

    print(f"[monitor] DONE. converged={converged} stop_block={stop_block} stop_time_ns={stop_time_ns}")
    print(f"[monitor] reason: {stop_reason}")

    if converged and process_alive(MDRUN_PID):
        print(f"[monitor] convergence reached - sending SIGTERM to mdrun pid {MDRUN_PID}")
        try:
            os.kill(MDRUN_PID, signal.SIGTERM)
        except OSError as e:
            print(f"[monitor] failed to signal mdrun pid {MDRUN_PID}: {e}")


if __name__ == "__main__":
    main()
