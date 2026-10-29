"""Protein-side WaterFP, contacts and SASA around a bound ligand (``oneopes protein``)."""

import hashlib
import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from MDAnalysis import AtomGroup, Universe

from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    ConvergenceSummary,
    Tolerances,
    compare_blocks,
    stack_blocks,
    summarize,
    times_table,
    write_block_tables,
    write_summary,
)
from oneopes.system import SystemSelection, select_system
from oneopes.trajectory_blocks import BLOCK_NS_DEFAULT, FrameLayout, frame_layout
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

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    candidate_cutoff_A: float = CANDIDATE_CUTOFF_A_DEFAULT
    block_ns: float = BLOCK_NS_DEFAULT
    first_frame: int = 0
    last_frame: int | None = None
    ligand_selection: str | None = None
    water_selection: str | None = None
    tolerances: Tolerances = DEFAULT_TOLERANCES


@dataclass(frozen=True)
class ProteinSystem:
    """A protein-ligand system: its trajectory, its complex structure, and which atoms are which."""
    trajectory: Universe
    """Topology and production trajectory."""
    complex: Universe
    """Topology and complex coordinates, for the candidate pool and SASA."""
    selection: SystemSelection
    chain_ids: np.ndarray | None
    """Chain IDs as written in the complex file, one per atom; None when unknown."""


def protein_system(trajectory: Universe, complex: Universe, complex_file: Universe | None = None,
                   ligand_selection: str | None = None, water_selection: str | None = None) -> ProteinSystem:
    """Check that the inputs describe the same atoms, and find the protein, ligand and water.

    ``complex_file`` is the complex file read on its own: its protein and ligand heavy-atom names must
    match the topology (water and hydrogen naming may differ between formats), and its chain IDs fill
    ``Chain_ID``.
    """
    validate_same_atoms(trajectory, complex, "complex vs trajectory topology")
    selection = select_system(trajectory, ligand_selection, water_selection)
    chain_ids = None
    if complex_file is not None:
        validate_same_atoms(trajectory, complex_file, "complex structure",
                            indices=np.concatenate([selection.protein_heavy, selection.ligand_heavy]))
        chain_ids = structure_chain_ids(complex_file)
    return ProteinSystem(trajectory=trajectory, complex=complex, selection=selection, chain_ids=chain_ids)


@dataclass(frozen=True)
class FrameAnalysis:
    """WaterFP and Contact/Bridge occupancy of the candidate atoms over one range of frames."""
    waterfp: WaterFP
    contacts: Contacts


def analyse_frames(system: ProteinSystem, candidate_atoms: AtomGroup, frames: range) -> FrameAnalysis:
    atoms = system.trajectory.atoms
    ligand, water = atoms[system.selection.ligand_heavy], atoms[system.selection.water_oxygens]
    return FrameAnalysis(
        waterfp=WaterFP(candidate_atoms, water).run(start=frames.start, stop=frames.stop),
        contacts=Contacts(candidate_atoms, ligand, water).run(start=frames.start, stop=frames.stop),
    )


def contacts_per_block(contacts: Sequence[Contacts], times_ns: Sequence[tuple[float, float]],
                       labels: Sequence[str]) -> pd.DataFrame:
    """One row per (block, candidate atom): block, t_start_ns, t_end_ns, then the Contacts table."""
    times = times_table(times_ns).drop(columns="t_mid_ns")
    return times.merge(stack_blocks([block.table(labels) for block in contacts]), on="block")


def write_outputs(outdir: Path, candidates: pd.DataFrame, layout: FrameLayout, blocks: Sequence[FrameAnalysis],
                  full: FrameAnalysis, sasa: np.ndarray, summary: ConvergenceSummary) -> None:
    """Write the candidate pool, the block-wise tables, the convergence summary and the final per-atom table."""
    labels = list(candidates["Atom"])
    candidates.to_csv(outdir / "candidate_atom_mapping.csv", index=False)
    write_block_tables(outdir, [block.waterfp for block in blocks], layout.block_times_ns, summary.tolerances, labels)
    contacts_per_block([block.contacts for block in blocks], layout.block_times_ns, labels).to_csv(
        outdir / "contact_occupancy_per_block.csv", index=False)
    write_summary(outdir / "convergence_summary.json", summary,
                  trajectory_start_ns=layout.trajectory_start_ns, trajectory_end_ns=layout.trajectory_end_ns,
                  analysis_start_ns=layout.analysis_start_ns, analysis_end_ns=layout.analysis_end_ns)
    table = build_final_table(candidates, full.waterfp.results.fp, full.contacts.results.contact_occupancy,
                              full.contacts.results.bridge_occupancy, sasa, layout.analysis_time_ns,
                              summary.converged_ns)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))


def run_analysis(system: ProteinSystem, options: AnalysisOptions, outdir: Path) -> dict:
    """Analyse every complete block and the whole analysed range of an existing trajectory.

    Every transition is scored whether or not convergence is reached; the final per-atom values
    cover the whole analysed range. Writes all outputs to ``outdir`` and returns the run's metadata.
    """
    sel = system.selection
    logger.info("ligand: %s (%d heavy atoms); water: %s (%d oxygens)", sel.ligand_description,
                len(sel.ligand_heavy), sel.water_description, len(sel.water_oxygens))
    candidates = select_candidates(system.complex, sel, options.candidate_cutoff_A, system.chain_ids)
    candidate_index = candidates["Atom_index"].to_numpy()
    logger.info("%d candidate protein heavy atoms within %s A", len(candidates), options.candidate_cutoff_A)
    layout = frame_layout(system.trajectory, options.block_ns, options.first_frame, options.last_frame)
    logger.info("%s", layout.describe())

    candidate_atoms = system.trajectory.atoms[candidate_index]
    blocks = [analyse_frames(system, candidate_atoms, frames) for frames in layout.blocks]
    full = analyse_frames(system, candidate_atoms, layout.analysed)
    stabilities = compare_blocks([block.waterfp.results for block in blocks], options.tolerances)
    summary = summarize(stabilities, layout.block_times_ns, options.block_ns, options.tolerances)
    logger.info("first sustained convergence: %s ns", summary.converged_ns)

    sasa = static_sasa(system.complex, sel.protein_all, sel.ligand_all, candidate_index)
    outdir.mkdir(parents=True, exist_ok=True)
    write_outputs(outdir, candidates, layout, blocks, full, sasa, summary)
    return {
        "candidate_cutoff_A": options.candidate_cutoff_A,
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "bridge_water_cutoff_A": BRIDGE_WATER_CUTOFF_A,
        "n_candidate_atoms": len(candidates),
        "ligand_selection": sel.ligand_description,
        "water_selection": sel.water_description,
        "frames": asdict(layout),
    }


def json_value(value: object) -> object:
    """JSON stand-in for the values ``json`` cannot write: a range as [start, stop], numpy numbers as floats."""
    return [value.start, value.stop] if isinstance(value, range) else float(value)


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
    try:
        u_traj = Universe(str(topology), str(trajectory))
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and trajectory are incompatible: {err}") from err
    try:
        u_complex = Universe(str(topology), str(complex))
        complex_file = Universe(str(complex))
    except Exception as err:
        raise SystemExit(f"[protein-waterfp] topology and complex are incompatible: {err}") from err

    options = AnalysisOptions(
        candidate_cutoff_A=candidate_cutoff_A, block_ns=block_ns, first_frame=first_frame, last_frame=last_frame,
        ligand_selection=ligand_selection, water_selection=water_selection, tolerances=tol,
    )
    system = protein_system(u_traj, u_complex, complex_file, ligand_selection, water_selection)
    meta = run_analysis(system, options, output_dir)
    inputs = {"complex": complex, "topology": topology, "trajectory": trajectory}
    meta["inputs"] = {name: {"file": path.name, "sha256": _sha256(path), "bytes": path.stat().st_size}
                      for name, path in inputs.items()}
    meta["options"] = asdict(options)
    meta["versions"] = _versions()
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=json_value))
    logger.info("%d candidates -> %s", meta["n_candidate_atoms"], output_dir / "final_per_atom_table.csv")
