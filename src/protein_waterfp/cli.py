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
from convergence import DEFAULT_TOLERANCES, compare_blocks, summarize, write_block_tables, write_summary
from trajectory_blocks import complete_blocks, frame_step_ps, frame_time_ns, frames_per_block, time_span_ns
from waterfp import WaterFP

from . import __version__
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


def _contacts_per_block(contacts: list[StructuralBlock], block_times: list[tuple[float, float]],
                        labels: list[str]) -> pd.DataFrame:
    """contact_occupancy_per_block.csv: one row per (block, candidate atom)."""
    tables = [pd.DataFrame({"block": b, "t_start_ns": start, "t_end_ns": end, "atom": labels,
                            "Contact_occupancy_5A": st.contact_occupancy, "Bridge_occupancy": st.bridge_occupancy,
                            "n_frames": st.n_frames})
              for b, (st, (start, end)) in enumerate(zip(contacts, block_times))]
    return pd.concat(tables, ignore_index=True)


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
    block_times = [time_span_ns(u_traj, frames, dt_ps) for frames in block_ranges]
    traj_start_ns, traj_end_ns = frame_time_ns(u_traj, 0), frame_time_ns(u_traj, n_frames - 1)
    analysis_start_ns, analysis_end_ns = frame_time_ns(u_traj, first_frame), frame_time_ns(u_traj, end_frame - 1)
    trajectory_length_ns = traj_end_ns - traj_start_ns
    final_analysis_time_ns = analysis_end_ns - analysis_start_ns
    log(f"[protein-waterfp] trajectory: {n_frames} frames, dt {dt_ps} ps, {traj_start_ns:g}-{traj_end_ns:g} ns "
        f"({trajectory_length_ns:g} ns); analysed frames {first_frame}-{end_frame - 1} ({final_analysis_time_ns:g} ns)")
    log(f"[protein-waterfp] {len(block_ranges)} blocks of {block_length} frames ({options.block_ns} ns), "
        f"{len(block_ranges) - 1} transitions; {unused_tail_frames} trailing frame(s) not in a complete block")

    # Compute: WaterFP and contacts of each block, then of the whole analysed range.
    candidate_atoms = u_traj.atoms[cand_idx]
    water_oxygens = u_traj.atoms[sel.water_oxygens]
    blocks, contacts = [], []
    for b, (frames, (start_ns, end_ns)) in enumerate(zip(block_ranges, block_times)):
        blocks.append(WaterFP(candidate_atoms, water_oxygens).run(start=frames.start, stop=frames.stop))
        contacts.append(structural_block(u_traj, frames.start, frames.stop, cand_idx, sel.ligand_heavy,
                                         sel.water_oxygens))
        log(f"[protein-waterfp] block {b} ({start_ns:g}-{end_ns:g} ns) done")
    log(f"[protein-waterfp] full-trajectory pass over frames {first_frame}-{end_frame - 1}")
    fp_full = WaterFP(candidate_atoms, water_oxygens).run(start=first_frame, stop=end_frame).results.fp
    st_full = structural_block(u_traj, first_frame, end_frame, cand_idx, sel.ligand_heavy, sel.water_oxygens)

    # Judge: convergence over the blocks.
    tol = DEFAULT_TOLERANCES
    stabilities = compare_blocks([block.results for block in blocks], tol)
    summary = summarize(stabilities, block_times, options.block_ns, tol)
    log(f"[protein-waterfp] first sustained convergence: {summary.converged_ns} ns")

    # Report: per-block tables, convergence summary and the final per-atom table.
    write_block_tables(outdir, blocks, block_times, tol, labels)
    _contacts_per_block(contacts, block_times, labels).to_csv(outdir / "contact_occupancy_per_block.csv", index=False)
    write_summary(outdir / "convergence_summary.json", summary,
                  trajectory_start_ns=traj_start_ns, trajectory_end_ns=traj_end_ns,
                  analysis_start_ns=analysis_start_ns, analysis_end_ns=analysis_end_ns)

    sasa = static_sasa(u_complex, sel.protein_all, sel.ligand_all, cand_idx)
    table = build_final_table(candidates, fp_full, st_full.contact_occupancy, st_full.bridge_occupancy,
                              sasa, final_analysis_time_ns, summary.converged_ns)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))

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
    print(f"[protein-waterfp] {meta['n_candidate_atoms']} candidates -> {outdir / 'final_per_atom_table.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
