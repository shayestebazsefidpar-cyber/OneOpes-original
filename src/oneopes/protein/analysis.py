"""Protein-side WaterFP, contacts and SASA around a bound ligand (``oneopes protein``)."""

import logging
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import pandas as pd
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number

from oneopes.block_run import BlockOptions, block_run, run_summary
from oneopes.protein.candidates import CANDIDATE_CUTOFF_A_DEFAULT, candidate_details, select_candidates
from oneopes.protein.contacts import BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, measure_contacts
from oneopes.protein.final_table import build_final_table, final_table_markdown
from oneopes.protein.inputs import ProteinSystem, protein_system
from oneopes.protein.sasa import static_sasa
from oneopes.provenance import provenance
from oneopes.reports import write_run
from oneopes.system import SystemOptions, load_universe
from oneopes.trajectory_blocks import FrameLayout, FrameWindow, frame_layout
from oneopes.waterfp import HydrationOptions, measure_hydration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    """Settings of one protein analysis; also the options of ``oneopes protein``."""
    candidate_cutoff_A: Annotated[float, Parameter(name="--candidate-cutoff-A", validator=Number(gt=0))] = (
        CANDIDATE_CUTOFF_A_DEFAULT)
    """Largest distance (A) from the ligand of a candidate protein atom."""
    system: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions()
    window: Annotated[FrameWindow, Parameter(name="*")] = FrameWindow()
    hydration: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions()
    blocks: Annotated[BlockOptions, Parameter(name="*")] = BlockOptions()


def measure_frames(system: ProteinSystem, candidates: pd.DataFrame, frames: range,
                   hydration: HydrationOptions) -> xr.Dataset:
    """Hydration (:func:`measure_hydration`) and contacts (:func:`measure_contacts`) of the candidates over
    ``frames``, along ``atom`` (named by their ``Atom`` label), with their details as coordinates."""
    u = system.trajectory
    candidate_atoms, labels = u.atoms[candidates["Atom_index"].to_numpy()], list(candidates["Atom"])
    ligand, water = system.selection.ligand(u), system.selection.water(u)
    measured = xr.merge([measure_hydration(candidate_atoms, water, frames, hydration, labels),
                         measure_contacts(candidate_atoms, ligand, water, frames, labels)])
    return measured.assign_coords(candidate_details(candidates).data_vars)


@dataclass(frozen=True)
class ProteinAnalysis:
    """Everything one protein analysis computed."""
    system: ProteinSystem
    options: AnalysisOptions
    candidates: pd.DataFrame
    """The candidate pool (see :func:`select_candidates`), in the order of every per-atom result."""
    layout: FrameLayout
    blocks: xr.Dataset
    """The :func:`block_run` of the :func:`measure_frames` of each block, with the frame layout's times."""
    full: xr.Dataset
    """The :func:`measure_frames` of the whole analysed range, with the static ``sasa`` (A^2) of each atom."""


def run_analysis(system: ProteinSystem, options: AnalysisOptions) -> ProteinAnalysis:
    """Analyse every complete block and the whole analysed range of an existing trajectory.

    Every transition is scored whether or not convergence is reached; the final per-atom values
    cover the whole analysed range.
    """
    sel = system.selection
    logger.info("ligand: %s (%d heavy atoms); water: %s (%d oxygens)", sel.ligand_selection,
                len(sel.ligand_heavy), sel.water_selection, len(sel.water_oxygens))
    candidates = select_candidates(system, options.candidate_cutoff_A)
    logger.info("%d candidate protein heavy atoms within %s A", len(candidates), options.candidate_cutoff_A)
    layout = frame_layout(system.trajectory, options.blocks.block_ns, options.window)
    logger.info("%s", layout.describe())

    measured = [block.label(measure_frames(system, candidates, block.frames, options.hydration))
                for block in layout.blocks]
    blocks = block_run(measured, options.blocks, options.hydration,
                       trajectory_start_ns=layout.trajectory_start_ns, trajectory_end_ns=layout.trajectory_end_ns,
                       analysis_start_ns=layout.analysis_start_ns, analysis_end_ns=layout.analysis_end_ns)
    full = measure_frames(system, candidates, layout.analysed, options.hydration)
    full = full.assign(sasa=("atom", static_sasa(system, full.atom_index.values)))
    logger.info("first sustained convergence: %s ns", run_summary(blocks).converged_ns)
    return ProteinAnalysis(system=system, options=options, candidates=candidates, layout=layout, blocks=blocks,
                           full=full)


OCCUPANCY_COLUMNS = {"contact_occupancy": "Contact_occupancy_5A", "bridge_occupancy": "Bridge_occupancy"}
"""The occupancies' column names in the per-block table."""


def write_outputs(outdir: Path, analysis: ProteinAnalysis, metadata: Mapping[str, Any] | None = None) -> None:
    """Write the candidate pool, the block run (blocks.nc, block-wise tables, convergence summary, and
    with ``metadata`` run_metadata.json) with the contacts per block, and the final per-atom table."""
    candidates = analysis.candidates
    outdir.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(outdir / "candidate_atom_mapping.csv", index=False)
    write_run(outdir, analysis.blocks, metadata)
    contacts = analysis.blocks[list(OCCUPANCY_COLUMNS)].to_dataframe().rename(columns=OCCUPANCY_COLUMNS)
    contacts.to_csv(outdir / "contact_occupancy_per_block.csv")
    table = build_final_table(analysis.full)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))


def analysis_details(analysis: ProteinAnalysis) -> dict[str, Any]:
    """What the analysis used: cutoffs, selections, frame layout and options."""
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
    }


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
    output_dir
        Directory for blocks.nc, the tables, the final per-atom table and run_metadata.json.
    """
    system = protein_system(load_universe(topology, trajectory), load_universe(topology, complex),
                            load_universe(complex), options.system)
    analysis = run_analysis(system, options)
    inputs = {"complex": complex, "topology": topology, "trajectory": trajectory}
    write_outputs(output_dir, analysis, provenance(inputs) | analysis_details(analysis))
    logger.info("%d candidates -> %s", len(analysis.candidates), output_dir / "final_per_atom_table.csv")
