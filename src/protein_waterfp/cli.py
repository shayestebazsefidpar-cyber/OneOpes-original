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
final_per_atom_table.csv (+ a readable final_per_atom_table.md).
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
from trajectory_blocks import complete_blocks, frame_step_ps, frames_per_block
from convergence import (
    DEFAULT_TOLERANCES,
    StabilityResult,
    compare_blocks,
    first_convergence,
    stable_in_a_row,
    transitions_table,
)
from waterfp import WaterFP

from . import __version__
from .convergence import fp_ranks
from .final_table import build_final_table, final_table_markdown
from .sasa import static_sasa
from .selection import (
    CANDIDATE_CUTOFF_A_DEFAULT,
    select_candidates,
    select_system,
    structure_chain_ids,
    validate_same_atoms,
)
from .trajectory import BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, StructuralBlock, structural_block

BLOCK_NS_DEFAULT = 5.0


@dataclass(frozen=True)
class AnalysisOptions:
    candidate_cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT
    block_ns: float = BLOCK_NS_DEFAULT
    first_frame: int = 0
    last_frame: int | None = None
    ligand_selection: str | None = None
    water_selection: str | None = None


def _after_convergence(stabilities: list[StabilityResult], converged_at: int | None) -> dict:
    """How the transitions after the first convergence behaved."""
    if converged_at is None:
        return {"n_transitions_after_first_convergence": 0,
                "n_unstable_transitions_after_first_convergence": 0,
                "remained_stable_after_first_convergence": None}
    later = stabilities[converged_at + 1:]
    unstable = sum(not s.passed for s in later)
    return {"n_transitions_after_first_convergence": len(later),
            "n_unstable_transitions_after_first_convergence": unstable,
            "remained_stable_after_first_convergence": unstable == 0}


def _frame_time_ns(u, frame: int) -> float:
    return float(u.trajectory[frame].time) / 1000.0


def _fp_per_block(blocks: list[WaterFP], block_times: list[tuple[float, float]], labels: list[str]) -> pd.DataFrame:
    """fp_values_per_block.csv: one row per (block, candidate atom)."""
    tables = [block.fp_table(labels).assign(block=b, t_start_ns=start, t_end_ns=end, t_mid_ns=0.5 * (start + end))
              for b, (block, (start, end)) in enumerate(zip(blocks, block_times))]
    return pd.concat(tables, ignore_index=True)[["block", "t_start_ns", "t_end_ns", "t_mid_ns", "atom", "FP"]]


def _rdf_per_block(blocks: list[WaterFP], labels: list[str]) -> pd.DataFrame:
    """rdf_profiles_per_block.csv: one row per (block, candidate atom, r)."""
    tables = [block.profile_table(labels).assign(block=b) for b, block in enumerate(blocks)]
    return pd.concat(tables, ignore_index=True)[["block", "atom", "r_nm", "g_r"]]


def _contacts_per_block(contacts: list[StructuralBlock], block_times: list[tuple[float, float]],
                        labels: list[str]) -> pd.DataFrame:
    """contact_occupancy_per_block.csv: one row per (block, candidate atom)."""
    tables = [pd.DataFrame({"block": b, "t_start_ns": start, "t_end_ns": end, "atom": labels,
                            "Contact_occupancy_5A": st.contact_occupancy, "Bridge_occupancy": st.bridge_occupancy,
                            "n_frames": st.n_frames})
              for b, (st, (start, end)) in enumerate(zip(contacts, block_times))]
    return pd.concat(tables, ignore_index=True)


def _write_summary_txt(path: Path, summary: dict, transitions: pd.DataFrame) -> None:
    """The human-readable convergence summary: the JSON summary plus one line per transition."""
    s = summary
    with open(path, "w") as fh:
        fh.write("Protein-side WaterFP retrospective block-wise convergence\n" + "=" * 57 + "\n")
        fh.write(f"Trajectory: {s['trajectory_start_ns']:g}-{s['trajectory_end_ns']:g} ns "
                 f"({s['trajectory_length_ns']:g} ns, {s['n_frames']} frames, dt {s['dt_ps']:g} ps)\n")
        fh.write(f"Blocks: {s['number_of_blocks']} x {s['block_size_ns']} ns, {s['number_of_transitions']} transitions; "
                 f"{s['frames_not_in_a_complete_block']} trailing frame(s) not in a complete block\n")
        fh.write(f"Thresholds: FP rel. change <= {s['fp_threshold']}, RDF nRMSD <= {s['rdf_nrmsd_threshold']}, "
                 f"Spearman >= {s['spearman_threshold']}; {s['stable_transitions_required']} consecutive stable "
                 "transitions\n")
        fh.write(f"Convergence reached: {s['convergence_reached']}\n")
        fh.write(f"First sustained convergence: {s['first_sustained_convergence_ns']} ns "
                 f"(transition {s['first_sustained_convergence_transition']})\n")
        fh.write(f"After first convergence: {s['n_transitions_after_first_convergence']} transitions, "
                 f"{s['n_unstable_transitions_after_first_convergence']} unstable\n")
        fh.write(f"Final values over {s['final_analysis_start_ns']:g}-{s['final_analysis_end_ns']:g} ns "
                 f"({s['final_analysis_time_ns']:g} ns)\n")
        for row in transitions.to_dict("records"):
            fh.write(f"  {row['transition']:>7} {row['block_start_ns']:g}-{row['block_end_ns']:g} ns  "
                     f"dFP={row['FP_relative_change']:.3f} nRMSD={row['RDF_nRMSD']:.3f} "
                     f"rho={row['Spearman_rho']:.4f} all_pass={row['all_criteria_pass']} "
                     f"count={row['stable_transition_count']} reached={row['convergence_reached']}\n")


def run_analysis(u_complex, u_traj, options: AnalysisOptions, outdir, log=print, complex_structure=None) -> dict:
    """Retrospective analysis of an existing trajectory; writes all outputs to ``outdir``.

    Every complete block is analysed and every transition scored, whether or not
    convergence is reached; final per-atom values cover the whole analysed range.
    ``complex_structure``: the complex file read on its own; its protein and ligand
    atom names are checked and its chain IDs fill ``Chain_ID`` (empty when None).
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    validate_same_atoms(u_traj, u_complex, "complex vs trajectory topology")

    sel = select_system(u_traj, options.ligand_selection, options.water_selection)
    log(f"[protein-waterfp] ligand: {sel.ligand_description} ({len(sel.ligand_heavy)} heavy atoms); "
        f"water: {sel.water_description} ({len(sel.water_oxygens)} oxygens)")

    chain_ids = None
    if complex_structure is not None:
        # water/hydrogen naming may legitimately differ between file formats: check protein and ligand heavy atoms
        validate_same_atoms(u_traj, complex_structure, "complex structure",
                            indices=np.concatenate([sel.protein_heavy, sel.ligand_heavy]))
        chain_ids = structure_chain_ids(complex_structure)
    candidates = select_candidates(u_complex, sel, options.candidate_cutoff_A, chain_ids)
    labels = list(candidates["Atom"])
    cand_idx = candidates["Atom_index"].to_numpy()
    candidates.to_csv(outdir / "candidate_atom_mapping.csv", index=False)
    log(f"[protein-waterfp] {len(candidates)} candidate protein heavy atoms within {options.candidate_cutoff_A} A")

    n_frames = len(u_traj.trajectory)
    dt_ps = frame_step_ps(u_traj)
    block_length = frames_per_block(options.block_ns, dt_ps)
    first_frame = options.first_frame
    end_frame = n_frames if options.last_frame is None else options.last_frame
    if not 0 <= first_frame < end_frame <= n_frames:
        raise ValueError(f"Invalid frame range [{first_frame}, {end_frame}) for a {n_frames}-frame trajectory.")
    block_ranges = complete_blocks(first_frame, end_frame, block_length)
    if not block_ranges:
        raise ValueError(f"Frames [{first_frame}, {end_frame}) hold less than one {options.block_ns} ns block.")
    unused_tail_frames = end_frame - block_ranges[-1].stop
    block_len_ns = block_length * dt_ps / 1000.0
    traj_start_ns, traj_end_ns = _frame_time_ns(u_traj, 0), _frame_time_ns(u_traj, n_frames - 1)
    analysis_start_ns, analysis_end_ns = _frame_time_ns(u_traj, first_frame), _frame_time_ns(u_traj, end_frame - 1)
    trajectory_length_ns = traj_end_ns - traj_start_ns
    final_analysis_time_ns = analysis_end_ns - analysis_start_ns
    log(f"[protein-waterfp] trajectory: {n_frames} frames, dt {dt_ps} ps, {traj_start_ns:g}-{traj_end_ns:g} ns "
        f"({trajectory_length_ns:g} ns); analysed frames {first_frame}-{end_frame - 1} ({final_analysis_time_ns:g} ns)")
    log(f"[protein-waterfp] {len(block_ranges)} blocks of {block_length} frames ({options.block_ns} ns), "
        f"{len(block_ranges) - 1} transitions; {unused_tail_frames} trailing frame(s) not in a complete block")

    # Compute: WaterFP and contacts of each block, then of the whole analysed range.
    candidate_atoms = u_traj.atoms[cand_idx]
    water_oxygens = u_traj.atoms[sel.water_oxygens]
    blocks, contacts, block_times = [], [], []
    for b, frames in enumerate(block_ranges):
        blocks.append(WaterFP(candidate_atoms, water_oxygens).run(start=frames.start, stop=frames.stop))
        contacts.append(structural_block(u_traj, frames.start, frames.stop, cand_idx, sel.ligand_heavy,
                                         sel.water_oxygens))
        t_start = _frame_time_ns(u_traj, frames.start)
        block_times.append((t_start, t_start + block_len_ns))
        log(f"[protein-waterfp] block {b} ({t_start:g}-{t_start + block_len_ns:g} ns) done")
    log(f"[protein-waterfp] full-trajectory pass over frames {first_frame}-{end_frame - 1}")
    fp_full = WaterFP(candidate_atoms, water_oxygens).run(start=first_frame, stop=end_frame).results.fp
    st_full = structural_block(u_traj, first_frame, end_frame, cand_idx, sel.ligand_heavy, sel.water_oxygens)

    # Judge: convergence over the blocks.
    tol = DEFAULT_TOLERANCES
    stabilities = compare_blocks([block.results for block in blocks], tol)
    converged_at = first_convergence(stabilities, tol.n_stable)  # transition index, or None
    first_converged_ns = None if converged_at is None else block_times[converged_at + 1][1]
    log(f"[protein-waterfp] first sustained convergence: {first_converged_ns} ns")

    # Report: per-block tables, convergence metrics, the final per-atom table and the summary.
    fp_df = _fp_per_block(blocks, block_times, labels)
    fp_df.to_csv(outdir / "fp_values_per_block.csv", index=False)
    _rdf_per_block(blocks, labels).to_csv(outdir / "rdf_profiles_per_block.csv", index=False)
    _contacts_per_block(contacts, block_times, labels).to_csv(outdir / "contact_occupancy_per_block.csv", index=False)
    transitions = transitions_table(stabilities, block_times, tol.n_stable)
    transitions.to_csv(outdir / "convergence_metrics.csv", index=False)
    wide = fp_df.pivot(index="atom", columns="block", values="FP").loc[labels]
    wide.to_csv(outdir / "fp_values_by_block_wide.csv")
    ranks = pd.DataFrame({b: fp_ranks(wide[b].to_dict(), labels) for b in wide.columns}).astype(int)
    ranks.index.name = "atom"
    ranks.to_csv(outdir / "fp_ranking_by_block_wide.csv")

    sasa = static_sasa(u_complex, sel.protein_all, sel.ligand_all, cand_idx)
    table = build_final_table(candidates, fp_full, st_full.contact_occupancy, st_full.bridge_occupancy,
                              sasa, final_analysis_time_ns, first_converged_ns)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))

    summary = {
        "trajectory_length_ns": trajectory_length_ns,
        "trajectory_start_ns": traj_start_ns,
        "trajectory_end_ns": traj_end_ns,
        "n_frames": n_frames,
        "dt_ps": dt_ps,
        "block_size_ns": options.block_ns,
        "number_of_blocks": len(block_ranges),
        "number_of_transitions": len(stabilities),
        "frames_not_in_a_complete_block": unused_tail_frames,
        "stable_transitions_required": tol.n_stable,
        "fp_threshold": tol.fp_rel_tol,
        "rdf_nrmsd_threshold": tol.rdf_nrmsd_tol,
        "spearman_threshold": tol.spearman_tol,
        "convergence_reached": converged_at is not None,
        "convergence_status": "NOT_CONVERGED" if converged_at is None else "CONVERGED",
        "first_sustained_convergence_ns": first_converged_ns,
        "first_sustained_convergence_transition":
            None if converged_at is None else f"{converged_at}->{converged_at + 1}",
        **_after_convergence(stabilities, converged_at),
        "stable_transitions_in_a_row_at_end": (stable_in_a_row(stabilities) or [0])[-1],
        "final_analysis_start_ns": analysis_start_ns,
        "final_analysis_end_ns": analysis_end_ns,
        "final_analysis_time_ns": final_analysis_time_ns,
        "criterion": "a transition is stable when FP relative change, RDF nRMSD and Spearman all pass; "
                     "sustained convergence = stable_transitions_required consecutive stable transitions. "
                     "Retrospective: all blocks are analysed regardless of convergence.",
        "n_candidate_atoms": len(labels),
    }
    (outdir / "convergence_summary.json").write_text(json.dumps(summary, indent=2))
    _write_summary_txt(outdir / "convergence_summary.txt", summary, transitions)

    return {
        "candidate_cutoff_A": options.candidate_cutoff_A,
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "bridge_water_cutoff_A": BRIDGE_WATER_CUTOFF_A,
        "n_candidate_atoms": len(labels),
        "ligand_selection": sel.ligand_description,
        "water_selection": sel.water_description,
        "frames": {"n_frames": n_frames, "dt_ps": dt_ps, "frames_per_block": block_length,
                   "analysed_frame_range": [first_frame, end_frame - 1],
                   "frames_not_in_a_complete_block": unused_tail_frames,
                   "blocks": [[frames.start, frames.stop] for frames in block_ranges]},
        "final_values": {"FP": "full analysed frame range", "Contact_occupancy_5A": "full analysed frame range",
                         "Bridge_occupancy": "full analysed frame range",
                         "SASA_A2": "static, complex structure", "FP_Rank": "1 = most negative FP"},
        "convergence": summary,
    }


def _sha256(path: Path) -> str:
    with open(path, "rb") as fh:
        return hashlib.file_digest(fh, "sha256").hexdigest()


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

    options = AnalysisOptions(
        candidate_cutoff_A=args.candidate_cutoff_A, block_ns=args.block_ns, first_frame=args.first_frame,
        last_frame=args.last_frame, ligand_selection=args.ligand_selection, water_selection=args.water_selection,
    )
    meta = run_analysis(u_complex, u_traj, options, outdir, complex_structure=raw_complex)
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
