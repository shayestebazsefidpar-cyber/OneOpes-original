"""Protein-side WaterFP, contacts and SASA around a bound ligand (``oneopes protein``)."""

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile

from oneopes.block_run import BlockOptions, block_run, convergence_summary, production_at
from oneopes.convergence import FpPrecision, check_every_block, production_estimate
from oneopes.errors import InputError
from oneopes.plots import write_charts
from oneopes.protein.candidates import CANDIDATE_CUTOFF_A_DEFAULT, select_candidates
from oneopes.protein.contacts import BRIDGE_WATER_CUTOFF_A, CONTACT_CUTOFF_A, measure_contacts
from oneopes.protein.final_table import TABLE_NAMES, build_final_table, final_table_markdown
from oneopes.protein.inputs import ProteinSystem, protein_system
from oneopes.protein.sasa import static_sasa
from oneopes.provenance import provenance
from oneopes.reports import write_run
from oneopes.system import SystemOptions, load_universe
from oneopes.trajectory_blocks import FrameLayout, FrameWindow, frame_layout, stack_blocks
from oneopes.waterfp import HydrationOptions, measure_hydration

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnalysisOptions:
    """Settings of one protein analysis."""
    candidate_cutoff_A: Annotated[float, Parameter(name="--candidate-cutoff-A")] = CANDIDATE_CUTOFF_A_DEFAULT
    """Largest distance (A) from the ligand of a candidate protein atom."""
    system_options: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions()
    window: Annotated[FrameWindow, Parameter(name="*")] = FrameWindow()
    hydration_options: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions()
    block_options: Annotated[BlockOptions, Parameter(name="*")] = BlockOptions()
    fp_rel_se_tol: float = 0.05
    """Largest FP standard error, relative to the FP, of a converged run."""

    def __post_init__(self) -> None:
        if not self.candidate_cutoff_A > 0:
            raise InputError(f"candidate_cutoff_A must be > 0, got {self.candidate_cutoff_A}.")
        if not self.fp_rel_se_tol > 0:
            raise InputError(f"fp_rel_se_tol must be > 0, got {self.fp_rel_se_tol}.")


def measure_frames(system: ProteinSystem, candidates: xr.Dataset, frames: range,
                   hydration_options: HydrationOptions) -> xr.Dataset:
    """Hydration and contacts of the candidates over ``frames``, with the candidates' coordinates."""
    u = system.trajectory
    candidate_atoms, labels = u.atoms[candidates.atom_index.values], list(candidates.atom.values)
    ligand, water = system.selection.ligand(u), system.selection.water(u)
    measured = xr.merge([measure_hydration(candidate_atoms, water, frames, hydration_options, labels),
                         measure_contacts(candidate_atoms, ligand, water, frames, labels)])
    return measured.assign_coords(candidates.coords)


@dataclass(frozen=True)
class ProteinAnalysis:
    """Everything one protein analysis computed."""
    system: ProteinSystem
    options: AnalysisOptions
    candidates: xr.Dataset
    layout: FrameLayout
    run: xr.Dataset
    """The block run, one measurement per block."""
    final: xr.Dataset
    """Each atom over the production blocks the last check found: FP with its error, contacts, and ``sasa`` (A^2)."""


def final_values(system: ProteinSystem, run: xr.Dataset, hydration_options: HydrationOptions) -> xr.Dataset:
    """FP with its error and the mean contacts over the production blocks the last check found, and static SASA."""
    production = production_at(run, int(run.block[-1]))
    contacts = production[["contact_occupancy", "bridge_occupancy"]].mean("block")
    final = xr.merge([production_estimate(production, hydration_options), contacts])
    return final.assign(sasa=("atom", static_sasa(system, final.atom_index.values)))


def run_analysis(system: ProteinSystem, options: AnalysisOptions) -> ProteinAnalysis:
    """Analyse and check every complete block of a finished trajectory; raises InputError for fewer than two."""
    sel = system.selection
    logger.info("ligand: %s (%d heavy atoms); water: %s (%d oxygens)", sel.ligand_selection,
                len(sel.ligand_heavy), sel.water_selection, len(sel.water_oxygens))
    candidates = select_candidates(system, options.candidate_cutoff_A)
    logger.info("%d candidate protein heavy atoms within %s A", candidates.sizes["atom"], options.candidate_cutoff_A)
    layout = frame_layout(system.trajectory, options.block_options.block_ns, options.window)
    logger.info("%s", layout.describe())
    if len(layout.blocks) < 2:
        raise InputError(f"The analysed frames hold one {options.block_options.block_ns} ns block; the jackknife "
                         "needs at least two: use a shorter --block-ns.")

    blocks = stack_blocks([block.label(measure_frames(system, candidates, block.frames, options.hydration_options))
                           for block in layout.blocks])
    goal = FpPrecision(options.fp_rel_se_tol)
    checks = check_every_block(blocks, options.block_options.criteria, options.hydration_options, goal)
    run = block_run(blocks, checks, options.block_options, options.hydration_options, goal,
                    trajectory_start_ns=layout.trajectory_start_ns, trajectory_end_ns=layout.trajectory_end_ns,
                    analysis_start_ns=layout.analysis_start_ns, analysis_end_ns=layout.analysis_end_ns)
    summary = convergence_summary(run)
    logger.info("production from %s ns (%s blocks); first sustained convergence: %s ns",
                summary.production_start_ns, summary.n_production_blocks, summary.converged_ns)
    final = final_values(system, run, options.hydration_options)
    return ProteinAnalysis(system=system, options=options, candidates=candidates, layout=layout, run=run, final=final)


CHARTED_ATOMS = 24
"""The charts show this many candidates: those with the most negative production FP."""


def write_outputs(outdir: Path, analysis: ProteinAnalysis, metadata: Mapping[str, Any] | None = None) -> None:
    """Write the candidate pool, the block run with its contacts per block, the final per-atom table, then charts."""
    outdir.mkdir(parents=True, exist_ok=True)
    candidate_table = analysis.candidates.to_dataframe().reset_index().rename(columns=TABLE_NAMES)
    candidate_table.to_csv(outdir / "candidate_atom_mapping.csv", index=False)
    write_run(outdir, analysis.run, metadata)
    contacts = analysis.run[["contact_occupancy", "bridge_occupancy"]].to_dataframe().reset_index()
    contacts.rename(columns=TABLE_NAMES).to_csv(outdir / "contact_occupancy_per_block.csv", index=False)
    table = build_final_table(analysis.final)
    table.to_csv(outdir / "final_per_atom_table.csv", index=False)
    (outdir / "final_per_atom_table.md").write_text(final_table_markdown(table))
    most_negative = analysis.final.fp.sortby(analysis.final.fp).atom.values[:CHARTED_ATOMS]
    write_charts(analysis.run.sel(atom=most_negative), outdir)


def analysis_details(analysis: ProteinAnalysis) -> dict[str, Any]:
    """What the analysis used: cutoffs, selections, frame layout and options."""
    sel = analysis.system.selection
    return {
        "contact_cutoff_A": CONTACT_CUTOFF_A,
        "bridge_water_cutoff_A": BRIDGE_WATER_CUTOFF_A,
        "n_candidate_atoms": analysis.candidates.sizes["atom"],
        "ligand_selection": sel.ligand_selection,
        "water_selection": sel.water_selection,
        "frames": analysis.layout,
        "options": analysis.options,
    }


def protein(*, complex_file: Annotated[ExistingFile, Parameter(name="--complex")], topology: ExistingFile,
            trajectory: ExistingFile, output_dir: Path,
            options: Annotated[AnalysisOptions, Parameter(name="*")] = AnalysisOptions()) -> None:
    """Analyse the protein atoms around a bound ligand: WaterFP, contacts and SASA, block by block.

    Post-processing of a finished unbiased trajectory. Candidates are the protein heavy atoms within
    --candidate-cutoff-A of the ligand in --complex; final_per_atom_table.csv gives their FP with its
    standard error, Contact and Bridge over the production blocks, and their SASA.

    Parameters
    ----------
    complex_file
        Protein-ligand structure used for the candidate pool and SASA.
    topology
        Topology with masses (e.g. a GROMACS .tpr).
    trajectory
        Production trajectory.
    output_dir
        Directory for blocks.nc, the tables, the final per-atom table, the charts (fp_vs_time.png,
        g_per_block.png) and run_metadata.json.
    """
    system = protein_system(load_universe(topology, trajectory), load_universe(topology, complex_file),
                            load_universe(complex_file), options.system_options)
    analysis = run_analysis(system, options)
    inputs = {"complex": complex_file, "topology": topology, "trajectory": trajectory}
    write_outputs(output_dir, analysis, provenance(inputs) | analysis_details(analysis))
    logger.info("%d candidates -> %s", analysis.candidates.sizes["atom"], output_dir / "final_per_atom_table.csv")
