"""
``protein-waterfp``: protein-side WaterFP analysis of one protein-ligand system.

All inputs are explicit (nothing is discovered from the directory layout)::

    protein-waterfp --complex complex.pdb --topology prod.tpr --trajectory prod.xtc \\
        --output-dir outputs/

Retrospective post-processing of an existing, completed unbiased MD
trajectory (its length is read from the input; nothing is stopped early).
Pipeline: candidate pool (protein heavy atoms within --candidate-cutoff-A of
the ligand, on --complex) -> consecutive --block-ns blocks over the whole
trajectory -> per block: WaterFP/RDF (ligand-side code), Contact, Bridge ->
convergence metrics for every transition (ligand-side criteria) and the first
sustained convergence -> full-trajectory FP/Contact/Bridge + static SASA ->
final_per_atom_table.csv.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .convergence import DEFAULT_TOLERANCES, ConvergenceTracker, block_waterfp, fp_ranks
from .final_table import build_final_table
from .sasa import static_sasa
from .selection import (
    CANDIDATE_CUTOFF_A_DEFAULT,
    select_candidates,
    select_system,
    structure_chain_ids,
    validate_same_atoms,
)
from .trajectory import BLOCK_NS_DEFAULT, BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, plan_blocks, structural_block, trajectory_dt_ps


@dataclass(frozen=True)
class AnalysisOptions:
    candidate_cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT
    block_ns: float = BLOCK_NS_DEFAULT
    first_frame: int = 0
    last_frame: int | None = None
    ligand_selection: str | None = None
    water_selection: str | None = None


def _write_csv(df: pd.DataFrame, path: Path, **kw) -> None:
    df.to_csv(path, index=False, **kw)


def _frame_time_ns(u, frame: int) -> float:
    return float(u.trajectory[frame].time) / 1000.0


def run_analysis(u_complex, u_traj, options: AnalysisOptions, outdir, log=print, chain_ids=None) -> dict:
    """Retrospective analysis of an existing trajectory; writes all outputs to ``outdir``.

    The trajectory length is read from ``u_traj``; every complete block is
    analysed and every transition scored, whether or not convergence has been
    reached. Final per-atom values cover the whole analysed frame range.
    ``chain_ids``: per-atom chain IDs read from the complex structure file
    (:func:`structure_chain_ids`); ``Chain_ID`` is left empty when None.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validate_same_atoms(u_traj, u_complex, "complex vs trajectory topology")

    sel = select_system(u_traj, options.ligand_selection, options.water_selection)
    log(f"[protein-waterfp] ligand: {sel.ligand_description} ({len(sel.ligand_heavy)} heavy atoms); "
        f"water: {sel.water_description} ({len(sel.water_oxygens)} oxygens)")

    if chain_ids is not None and len(chain_ids) != len(u_traj.atoms):
        raise ValueError(f"{len(chain_ids)} chain IDs for {len(u_traj.atoms)} atoms.")
    candidates = select_candidates(u_complex, sel, options.candidate_cutoff_A, chain_ids)
    labels = list(candidates["Atom"])
    cand_idx = candidates["Atom_index"].to_numpy()
    _write_csv(candidates, outdir / "candidate_atom_mapping.csv")
    log(f"[protein-waterfp] {len(candidates)} candidate protein heavy atoms within {options.candidate_cutoff_A} A")

    n_frames = len(u_traj.trajectory)
    dt_ps = trajectory_dt_ps(u_traj)
    plan = plan_blocks(n_frames, dt_ps, options.block_ns, options.first_frame, options.last_frame)
    block_len_ns = plan.frames_per_block * dt_ps / 1000.0
    traj_start_ns, traj_end_ns = _frame_time_ns(u_traj, 0), _frame_time_ns(u_traj, n_frames - 1)
    first, last = plan.first_frame, plan.last_frame_exclusive
    analysis_start_ns, analysis_end_ns = _frame_time_ns(u_traj, first), _frame_time_ns(u_traj, last - 1)
    trajectory_length_ns = traj_end_ns - traj_start_ns
    final_analysis_time_ns = analysis_end_ns - analysis_start_ns
    log(f"[protein-waterfp] trajectory: {n_frames} frames, dt {dt_ps} ps, {traj_start_ns:g}-{traj_end_ns:g} ns "
        f"({trajectory_length_ns:g} ns); analysed frames {first}-{last - 1} ({final_analysis_time_ns:g} ns)")
    log(f"[protein-waterfp] {len(plan.blocks)} blocks of {plan.frames_per_block} frames ({options.block_ns} ns), "
        f"{len(plan.blocks) - 1} transitions; {plan.unused_tail_frames} trailing frame(s) not in a complete block")

    tracker = ConvergenceTracker(tol=DEFAULT_TOLERANCES)
    fp_rows, contact_rows = [], []
    rdf_path = outdir / "rdf_profiles_per_block.csv"
    if rdf_path.exists():
        rdf_path.unlink()
    n = len(cand_idx)
    for b, (start, end) in enumerate(plan.blocks):
        t_start = _frame_time_ns(u_traj, start)
        t_end = t_start + block_len_ns
        fp, g, r_nm = block_waterfp(u_traj, start, end, cand_idx, sel.water_oxygens, labels)
        st = structural_block(u_traj, start, end, cand_idx, sel.ligand_heavy, sel.water_oxygens)
        tracker.add_block(b, t_start, t_end, fp, g, labels)
        for i, a in enumerate(labels):
            fp_rows.append({"block": b, "t_start_ns": t_start, "t_end_ns": t_end,
                            "t_mid_ns": 0.5 * (t_start + t_end), "atom": a, "FP": fp[a]})
            contact_rows.append({"block": b, "t_start_ns": t_start, "t_end_ns": t_end, "atom": a,
                                 "Contact_occupancy_5A": float(st.contact_occupancy[i]),
                                 "Bridge_occupancy": float(st.bridge_occupancy[i]), "n_frames": st.n_frames})
        rdf_block = pd.DataFrame({
            "block": b,
            "atom": np.repeat(labels, len(r_nm)),
            "r_nm": np.tile(r_nm, n),
            "g_r": np.concatenate([g[a] for a in labels]),
        })
        rdf_block.to_csv(rdf_path, mode="a", header=(b == 0), index=False)
        streak_txt = tracker.rows[-1]["stable_transition_count"] if tracker.rows else "-"
        log(f"[protein-waterfp] block {b} ({t_start:g}-{t_end:g} ns) done; stable transitions in a row: {streak_txt}")

    fp_df = pd.DataFrame(fp_rows)
    _write_csv(fp_df, outdir / "fp_values_per_block.csv")
    _write_csv(pd.DataFrame(contact_rows), outdir / "contact_occupancy_per_block.csv")
    _write_csv(pd.DataFrame(tracker.rows), outdir / "convergence_metrics.csv")
    wide = fp_df.pivot(index="atom", columns="block", values="FP").loc[labels]
    wide.to_csv(outdir / "fp_values_by_block_wide.csv")
    ranks = pd.DataFrame({b: fp_ranks(wide[b].to_dict(), labels) for b in wide.columns}).astype(int)
    ranks.index.name = "atom"
    ranks.to_csv(outdir / "fp_ranking_by_block_wide.csv")

    # Final values: the same WaterFP / Contact / Bridge code applied to the whole analysed range.
    log(f"[protein-waterfp] full-trajectory pass over frames {first}-{last - 1}")
    fp_full, _g_full, _r = block_waterfp(u_traj, first, last, cand_idx, sel.water_oxygens, labels)
    st_full = structural_block(u_traj, first, last, cand_idx, sel.ligand_heavy, sel.water_oxygens)

    sasa = static_sasa(u_complex, sel.protein_all, sel.ligand_all, cand_idx)
    table = build_final_table(
        candidates, fp_full,
        dict(zip(labels, st_full.contact_occupancy.tolist())),
        dict(zip(labels, st_full.bridge_occupancy.tolist())),
        sasa, final_analysis_time_ns, tracker.first_converged_ns,
    )
    _write_csv(table, outdir / "final_per_atom_table.csv")

    tol = tracker.tol
    first_block = tracker.first_converged_block
    summary = {
        "trajectory_length_ns": trajectory_length_ns,
        "trajectory_start_ns": traj_start_ns,
        "trajectory_end_ns": traj_end_ns,
        "n_frames": n_frames,
        "dt_ps": dt_ps,
        "block_size_ns": options.block_ns,
        "number_of_blocks": len(plan.blocks),
        "number_of_transitions": len(tracker.rows),
        "frames_not_in_a_complete_block": plan.unused_tail_frames,
        "stable_transitions_required": tol.n_stable,
        "fp_threshold": tol.fp_rel_tol,
        "rdf_nrmsd_threshold": tol.rdf_nrmsd_tol,
        "spearman_threshold": tol.spearman_tol,
        "convergence_reached": first_block is not None,
        "convergence_status": tracker.status,
        "first_sustained_convergence_ns": tracker.first_converged_ns,
        "first_sustained_convergence_transition": None if first_block is None else f"{first_block - 1}->{first_block}",
        **tracker.after_convergence(),
        "stable_transitions_in_a_row_at_end": tracker.streak,
        "final_analysis_start_ns": analysis_start_ns,
        "final_analysis_end_ns": analysis_end_ns,
        "final_analysis_time_ns": final_analysis_time_ns,
        "criterion": "a transition is stable when FP relative change, RDF nRMSD and Spearman all pass; "
                     "sustained convergence = stable_transitions_required consecutive stable transitions. "
                     "Retrospective: all blocks are analysed regardless of convergence.",
        "n_candidate_atoms": len(labels),
    }
    (outdir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    with open(outdir / "convergence_summary.txt", "w") as fh:
        fh.write("Protein-side WaterFP retrospective block-wise convergence\n" + "=" * 57 + "\n")
        fh.write(f"Trajectory: {traj_start_ns:g}-{traj_end_ns:g} ns ({trajectory_length_ns:g} ns, "
                 f"{n_frames} frames, dt {dt_ps:g} ps)\n")
        fh.write(f"Blocks: {len(plan.blocks)} x {options.block_ns} ns, {len(tracker.rows)} transitions; "
                 f"{plan.unused_tail_frames} trailing frame(s) not in a complete block\n")
        fh.write(f"Thresholds: FP rel. change <= {tol.fp_rel_tol}, RDF nRMSD <= {tol.rdf_nrmsd_tol}, "
                 f"Spearman >= {tol.spearman_tol}; {tol.n_stable} consecutive stable transitions\n")
        fh.write(f"Convergence reached: {summary['convergence_reached']}\n")
        fh.write(f"First sustained convergence: {summary['first_sustained_convergence_ns']} ns "
                 f"(transition {summary['first_sustained_convergence_transition']})\n")
        fh.write(f"After first convergence: {summary['n_transitions_after_first_convergence']} transitions, "
                 f"{summary['n_unstable_transitions_after_first_convergence']} unstable\n")
        fh.write(f"Final values over {analysis_start_ns:g}-{analysis_end_ns:g} ns ({final_analysis_time_ns:g} ns)\n")
        for row in tracker.rows:
            fh.write(f"  {row['transition']:>7} {row['block_start_ns']:g}-{row['block_end_ns']:g} ns  "
                     f"dFP={row['FP_relative_change']:.3f} nRMSD={row['RDF_nRMSD']:.3f} "
                     f"rho={row['Spearman_rho']:.4f} all_pass={row['all_criteria_pass']} "
                     f"count={row['stable_transition_count']} reached={row['convergence_reached']}\n")

    return {
        "candidate_cutoff_A": options.candidate_cutoff_A,
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "bridge_water_cutoff_A": BRIDGE_WATER_CUTOFF_A,
        "n_candidate_atoms": len(labels),
        "ligand_selection": sel.ligand_description,
        "water_selection": sel.water_description,
        "frames": {"n_frames": n_frames, "dt_ps": plan.dt_ps, "frames_per_block": plan.frames_per_block,
                   "analysed_frame_range": [first, last - 1],
                   "frames_not_in_a_complete_block": plan.unused_tail_frames,
                   "blocks": [list(bl) for bl in plan.blocks]},
        "final_values": {"FP": "full analysed frame range", "Contact_occupancy_5A": "full analysed frame range",
                         "Bridge_occupancy": "full analysed frame range",
                         "SASA_A2": "static, complex structure", "FP_Rank": "1 = most negative FP"},
        "convergence": summary,
    }


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _versions() -> dict:
    import importlib.metadata as md

    out = {"python": platform.python_version(), "protein_waterfp": __version__}
    for pkg in ("numpy", "pandas", "scipy", "MDAnalysis", "freesasa", "ligand-waterfp"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def parse_args(argv=None):
    p = argparse.ArgumentParser(prog="protein-waterfp", description=__doc__.split("\n\n")[0])
    p.add_argument("--complex", required=True, help="protein-ligand structure used for candidate selection and SASA")
    p.add_argument("--topology", required=True, help="topology with masses (e.g. GROMACS .tpr)")
    p.add_argument("--trajectory", required=True, help="production trajectory")
    p.add_argument("--output-dir", required=True)
    p.add_argument("--candidate-cutoff-A", type=float, default=CANDIDATE_CUTOFF_A_DEFAULT)
    p.add_argument("--block-ns", type=float, default=BLOCK_NS_DEFAULT)
    p.add_argument("--first-frame", type=int, default=0)
    p.add_argument("--last-frame", type=int, default=None, help="exclusive end frame (default: end of trajectory)")
    p.add_argument("--ligand-selection", default=None, help="MDAnalysis selection (default: auto-detect)")
    p.add_argument("--water-selection", default=None, help="MDAnalysis selection (default: known water names)")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    import MDAnalysis as mda

    inputs = {"complex": args.complex, "topology": args.topology, "trajectory": args.trajectory}
    for name, path in inputs.items():
        resolved = Path(path).resolve()
        if not resolved.is_file():
            raise SystemExit(f"[protein-waterfp] --{name} not found: {resolved}")
        print(f"[protein-waterfp] --{name} = {resolved}")
    outdir = Path(args.output_dir)
    print(f"[protein-waterfp] --output-dir = {outdir.resolve()}")

    try:
        u_traj = mda.Universe(args.topology, args.trajectory)
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and trajectory are incompatible: {err}") from err
    try:
        u_complex = mda.Universe(args.topology, args.complex)
        raw_complex = mda.Universe(args.complex)
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and complex are incompatible: {err}") from err
    # The complex supplies protein/ligand coordinates (candidate selection, SASA): check the
    # atom count, and the names of the protein and ligand heavy atoms (water/hydrogen naming
    # conventions may legitimately differ between file formats).
    sel = select_system(u_traj, args.ligand_selection, args.water_selection)
    validate_same_atoms(u_traj, raw_complex, "complex structure",
                        indices=np.concatenate([sel.protein_heavy, sel.ligand_heavy]))

    options = AnalysisOptions(
        candidate_cutoff_A=args.candidate_cutoff_A, block_ns=args.block_ns, first_frame=args.first_frame,
        last_frame=args.last_frame, ligand_selection=args.ligand_selection, water_selection=args.water_selection,
    )
    meta = run_analysis(u_complex, u_traj, options, outdir, chain_ids=structure_chain_ids(raw_complex))
    meta["inputs"] = {name: {"file": Path(p).name, "sha256": _sha256(Path(p)), "bytes": Path(p).stat().st_size}
                      for name, p in inputs.items()}
    meta["options"] = asdict(options)
    meta["versions"] = _versions()
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    conv = meta["convergence"]
    print(f"[protein-waterfp] {meta['n_candidate_atoms']} candidates; trajectory {conv['trajectory_length_ns']:g} ns, "
          f"{conv['final_analysis_time_ns']:g} ns analysed; "
          f"first sustained convergence: {conv['first_sustained_convergence_ns']} ns "
          f"-> {outdir / 'final_per_atom_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
