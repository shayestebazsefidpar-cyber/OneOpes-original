"""Protein-side WaterFP, contacts and SASA around a bound ligand (``oneopes protein``)."""

import hashlib
import json
import logging
import platform
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number
from MDAnalysis import AtomGroup, Universe

from oneopes import __version__
from oneopes.convergence import (
    DEFAULT_TOLERANCES,
    ConvergenceSummary,
    Tolerances,
    compare_blocks,
    summarize,
)
from oneopes.errors import InputError
from oneopes.protein.candidates import CANDIDATE_CUTOFF_A_DEFAULT, select_candidates
from oneopes.protein.contacts import BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, measure_contacts
from oneopes.protein.final_table import build_final_table, final_table_markdown
from oneopes.protein.inputs import ProteinSystem, protein_system
from oneopes.protein.sasa import static_sasa
from oneopes.reports import write_block_tables, write_summary
from oneopes.trajectory_blocks import (
    BLOCK_NS_DEFAULT,
    FrameLayout,
    frame_layout,
    label_block,
    stack_blocks,
)
from oneopes.waterfp import measure_hydration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    """Settings of one protein analysis; also the options of ``oneopes protein``."""
    candidate_cutoff_A: Annotated[float, Parameter(name="--candidate-cutoff-A", validator=Number(gt=0))] = (
        CANDIDATE_CUTOFF_A_DEFAULT)
    """Largest distance (A) from the ligand of a candidate protein atom."""
    block_ns: Annotated[float, Parameter(validator=Number(gt=0))] = BLOCK_NS_DEFAULT
    """Trajectory block size (ns)."""
    first_frame: int = 0
    """First analysed frame."""
    last_frame: int | None = None
    """Exclusive end frame (default: end of the trajectory)."""
    ligand_selection: str | None = None
    """MDAnalysis selection of the ligand (default: auto-detect)."""
    water_selection: str | None = None
    """MDAnalysis selection of the water (default: known water residue names)."""
    tolerances: Annotated[Tolerances, Parameter(name="*")] = DEFAULT_TOLERANCES


def measure_frames(system: ProteinSystem, candidate_atoms: AtomGroup, labels: Sequence[str],
                   frames: range) -> xr.Dataset:
    """Hydration (:func:`measure_hydration`) and contacts (:func:`measure_contacts`) of the candidates over
    ``frames``, along ``atom`` (named by ``labels``)."""
    atoms = system.trajectory.atoms
    ligand, water = atoms[system.selection.ligand_heavy], atoms[system.selection.water_oxygens]
    return xr.merge([measure_hydration(candidate_atoms, water, frames, labels=labels),
                     measure_contacts(candidate_atoms, ligand, water, frames, labels)])


@dataclass(frozen=True)
class ProteinAnalysis:
    """Everything one protein analysis computed."""
    system: ProteinSystem
    options: AnalysisOptions
    candidates: pd.DataFrame
    """The candidate pool (see :func:`select_candidates`), in the order of every per-atom result."""
    layout: FrameLayout
    blocks: xr.Dataset
    """The :func:`measure_frames` of each block, each compared with the one before (:func:`compare_blocks`)."""
    full: xr.Dataset
    """The :func:`measure_frames` of the whole analysed range."""
    sasa: np.ndarray
    summary: ConvergenceSummary


def run_analysis(system: ProteinSystem, options: AnalysisOptions) -> ProteinAnalysis:
    """Analyse every complete block and the whole analysed range of an existing trajectory.

    Every transition is scored whether or not convergence is reached; the final per-atom values
    cover the whole analysed range.
    """
    sel = system.selection
    logger.info("ligand: %s (%d heavy atoms); water: %s (%d oxygens)", sel.ligand_selection,
                len(sel.ligand_heavy), sel.water_selection, len(sel.water_oxygens))
    candidates = select_candidates(system, options.candidate_cutoff_A)
    candidate_index = candidates["Atom_index"].to_numpy()
    logger.info("%d candidate protein heavy atoms within %s A", len(candidates), options.candidate_cutoff_A)
    layout = frame_layout(system.trajectory, options.block_ns, options.first_frame, options.last_frame)
    logger.info("%s", layout.describe())

    candidate_atoms = system.trajectory.atoms[candidate_index]
    labels = list(candidates["Atom"])
    measured = [label_block(measure_frames(system, candidate_atoms, labels, block.frames), block)
                for block in layout.blocks]
    blocks = compare_blocks(stack_blocks(measured), options.tolerances)
    full = measure_frames(system, candidate_atoms, labels, layout.analysed)
    summary = summarize(blocks, options.block_ns, options.tolerances)
    logger.info("first sustained convergence: %s ns", summary.converged_ns)
    return ProteinAnalysis(system=system, options=options, candidates=candidates, layout=layout, blocks=blocks,
                           full=full, sasa=static_sasa(system, candidate_index), summary=summary)


OCCUPANCY_COLUMNS = {"contact_occupancy": "Contact_occupancy_5A", "bridge_occupancy": "Bridge_occupancy"}
"""The occupancies' column names in the tables."""


def final_values(full: xr.Dataset) -> pd.DataFrame:
    """One row per candidate atom, in order: FP, Contact_occupancy_5A, Bridge_occupancy."""
    table = full[["fp", *OCCUPANCY_COLUMNS]].to_dataframe().reset_index(drop=True)
    return table.rename(columns={"fp": "FP", **OCCUPANCY_COLUMNS})


def write_outputs(outdir: Path, analysis: ProteinAnalysis) -> None:
    """Write the candidate pool, the block-wise tables, the convergence summary and the final per-atom table."""
    candidates, layout = analysis.candidates, analysis.layout
    outdir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(outdir / "candidate_atom_mapping.csv", index=False)
    write_block_tables(outdir, analysis.blocks)
    contacts = analysis.blocks[list(OCCUPANCY_COLUMNS)].to_dataframe().rename(columns=OCCUPANCY_COLUMNS)
    contacts.to_csv(outdir / "contact_occupancy_per_block.csv")
    write_summary(outdir / "convergence_summary.json", analysis.summary,
                  trajectory_start_ns=layout.trajectory_start_ns, trajectory_end_ns=layout.trajectory_end_ns,
                  analysis_start_ns=layout.analysis_start_ns, analysis_end_ns=layout.analysis_end_ns)
    table = build_final_table(candidates, final_values(analysis.full), analysis.sasa)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))


def run_metadata(analysis: ProteinAnalysis) -> dict:
    """What the run used: cutoffs, selections, frame layout, options and package versions."""
    sel = analysis.system.selection
    return {
        "candidate_cutoff_A": analysis.options.candidate_cutoff_A,
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "bridge_water_cutoff_A": BRIDGE_WATER_CUTOFF_A,
        "n_candidate_atoms": len(analysis.candidates),
        "ligand_selection": sel.ligand_selection,
        "water_selection": sel.water_selection,
        "frames": asdict(analysis.layout),
        "options": asdict(analysis.options),
        "versions": {"python": platform.python_version(), "oneopes": __version__}
                    | {package: installed_version(package)
                       for package in ("numpy", "pandas", "scipy", "MDAnalysis", "freesasa")},
    }


def installed_version(package: str) -> str:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def json_value(value: range | np.number) -> list[int] | float:
    """JSON stand-in for the values ``json`` cannot write: a range as [start, stop], numpy numbers as floats."""
    return [value.start, value.stop] if isinstance(value, range) else float(value)


def file_record(path: Path) -> dict:
    """File name, SHA-256 and size of an input file."""
    with path.open("rb") as fh:
        sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    return {"file": path.name, "sha256": sha256, "bytes": path.stat().st_size}


def protein(*, complex: ExistingFile, topology: ExistingFile, trajectory: ExistingFile, output_dir: Path,
            options: Annotated[AnalysisOptions, Parameter(name="*")] = AnalysisOptions()) -> None:
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
    """
    try:
        u_traj = Universe(str(topology), str(trajectory))
    except Exception as err:
        raise InputError(f"topology and trajectory are incompatible: {err}") from err
    try:
        u_complex = Universe(str(topology), str(complex))
        complex_file = Universe(str(complex))
    except Exception as err:
        raise InputError(f"topology and complex are incompatible: {err}") from err

    system = protein_system(u_traj, u_complex, complex_file, options.ligand_selection, options.water_selection)
    analysis = run_analysis(system, options)
    write_outputs(output_dir, analysis)
    inputs = {"complex": complex, "topology": topology, "trajectory": trajectory}
    meta = run_metadata(analysis) | {"inputs": {name: file_record(path) for name, path in inputs.items()}}
    (output_dir / "run_metadata.json").write_text(json.dumps(meta, indent=2, default=json_value))
    logger.info("%d candidates -> %s", len(analysis.candidates), output_dir / "final_per_atom_table.csv")
