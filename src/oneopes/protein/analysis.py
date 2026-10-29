"""Protein-side WaterFP, contacts and SASA around a bound ligand (``oneopes protein``)."""

import hashlib
import json
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from cyclopts import Parameter
from cyclopts.types import ExistingFile

from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    Tolerances,
    compare_blocks,
    stack_blocks,
    summarize,
    times_table,
    write_block_tables,
    write_summary,
)
from oneopes.system import select_system
from oneopes.trajectory_blocks import (
    BLOCK_NS_DEFAULT,
    complete_blocks,
    frame_step_ps,
    frame_time_ns,
    frames_per_block,
    time_span_ns,
)
from oneopes.waterfp import WaterFP

from oneopes import __version__
from oneopes.protein.candidates import (
    CANDIDATE_CUTOFF_A_DEFAULT,
    select_candidates,
    structure_chain_ids,
    validate_same_atoms,
)
from oneopes.protein.contacts import BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, Contacts
from oneopes.protein.final_table import build_final_table, final_table_markdown
from oneopes.protein.sasa import static_sasa


@dataclass(frozen=True)
class AnalysisOptions:
    candidate_cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT
    block_ns: float = BLOCK_NS_DEFAULT
    first_frame: int = 0
    last_frame: int | None = None
    ligand_selection: str | None = None
    water_selection: str | None = None
    tolerances: Tolerances = DEFAULT_TOLERANCES


def contacts_per_block(contacts: Sequence[Contacts], times_ns: Sequence[tuple[float, float]],
                       labels: Sequence[str]) -> pd.DataFrame:
    """One row per (block, candidate atom): block, t_start_ns, t_end_ns, then the Contacts table."""
    times = times_table(times_ns).drop(columns="t_mid_ns")
    return times.merge(stack_blocks([block.table(labels) for block in contacts]), on="block")


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
    ligand = u_traj.atoms[sel.ligand_heavy]
    water_oxygens = u_traj.atoms[sel.water_oxygens]
    blocks, contacts = [], []
    for b, (frames, (start_ns, end_ns)) in enumerate(zip(block_ranges, block_times)):
        blocks.append(WaterFP(candidate_atoms, water_oxygens).run(start=frames.start, stop=frames.stop))
        contacts.append(Contacts(candidate_atoms, ligand, water_oxygens).run(start=frames.start, stop=frames.stop))
        log(f"[protein-waterfp] block {b} ({start_ns:g}-{end_ns:g} ns) done")
    log(f"[protein-waterfp] full-trajectory pass over frames {first_frame}-{end_frame - 1}")
    fp_full = WaterFP(candidate_atoms, water_oxygens).run(start=first_frame, stop=end_frame).results.fp
    contacts_full = Contacts(candidate_atoms, ligand, water_oxygens).run(start=first_frame, stop=end_frame).results

    # Judge: convergence over the blocks.
    tol = options.tolerances
    stabilities = compare_blocks([block.results for block in blocks], tol)
    summary = summarize(stabilities, block_times, options.block_ns, tol)
    log(f"[protein-waterfp] first sustained convergence: {summary.converged_ns} ns")

    # Report: per-block tables, convergence summary and the final per-atom table.
    write_block_tables(outdir, blocks, block_times, tol, labels)
    contacts_per_block(contacts, block_times, labels).to_csv(outdir / "contact_occupancy_per_block.csv", index=False)
    write_summary(outdir / "convergence_summary.json", summary,
                  trajectory_start_ns=traj_start_ns, trajectory_end_ns=traj_end_ns,
                  analysis_start_ns=analysis_start_ns, analysis_end_ns=analysis_end_ns)

    sasa = static_sasa(u_complex, sel.protein_all, sel.ligand_all, cand_idx)
    table = build_final_table(candidates, fp_full, contacts_full.contact_occupancy, contacts_full.bridge_occupancy,
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

    out = {"python": platform.python_version(), "oneopes": __version__}
    for pkg in ("numpy", "pandas", "scipy", "MDAnalysis", "freesasa"):
        try:
            out[pkg] = md.version(pkg)
        except md.PackageNotFoundError:
            out[pkg] = "unknown"
    return out


def protein(*, complex: ExistingFile, topology: ExistingFile, trajectory: ExistingFile, output_dir: Path,
            candidate_cutoff_A: Annotated[float, Parameter(name="--candidate-cutoff-A")] = CANDIDATE_CUTOFF_A_DEFAULT,
            first_frame: int = 0, last_frame: int | None = None,
            ligand_selection: str | None = None, water_selection: str | None = None,
            block_ns: float = BLOCK_NS_DEFAULT,
            tol: Annotated[Tolerances, Parameter(name="*")] = DEFAULT_TOLERANCES) -> None:
    """Analyse the protein atoms around a bound ligand: WaterFP, contacts and SASA, block by block.

    Retrospective post-processing of a completed unbiased trajectory (nothing is stopped
    early): candidate pool (protein heavy atoms within --candidate-cutoff-A of the ligand in
    --complex) -> consecutive blocks over the whole trajectory -> per block WaterFP, Contact
    and Bridge -> convergence of every transition -> full-trajectory FP, Contact and Bridge
    plus static SASA -> final_per_atom_table.csv (and a readable .md).

    Parameters
    ----------
    complex
        Protein-ligand structure used for the candidate pool and SASA.
    topology
        Topology with masses (e.g. a GROMACS .tpr).
    trajectory
        Production trajectory.
    candidate_cutoff_A
        Largest distance (A) from the ligand of a candidate protein atom.
    last_frame
        Exclusive end frame (default: end of the trajectory).
    ligand_selection
        MDAnalysis selection of the ligand (default: auto-detect).
    water_selection
        MDAnalysis selection of the water (default: known water residue names).
    block_ns
        Trajectory block size (ns).
    """
    import MDAnalysis as mda

    try:
        u_traj = mda.Universe(str(topology), str(trajectory))
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and trajectory are incompatible: {err}") from err
    try:
        u_complex = mda.Universe(str(topology), str(complex))
        raw_complex = mda.Universe(str(complex))
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and complex are incompatible: {err}") from err

    options = AnalysisOptions(
        candidate_cutoff_A=candidate_cutoff_A, block_ns=block_ns, first_frame=first_frame, last_frame=last_frame,
        ligand_selection=ligand_selection, water_selection=water_selection, tolerances=tol,
    )
    outdir = output_dir
    meta = run_analysis(u_complex, u_traj, options, outdir, complex_structure=raw_complex)
    inputs = {"complex": complex, "topology": topology, "trajectory": trajectory}
    meta["inputs"] = {name: {"file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                      for name, path in inputs.items()}
    meta["options"] = asdict(options)
    meta["versions"] = _versions()
    (outdir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=float))
    print(f"[protein-waterfp] {meta['n_candidate_atoms']} candidates -> {outdir / 'final_per_atom_table.csv'}")
