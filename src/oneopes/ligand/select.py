"""The G1/G2 selection commands: ``oneopes select`` over a frame window, ``oneopes select-block`` on a saved run."""
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import numpy as np
import xarray as xr
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number
from MDAnalysis.core.groups import AtomGroup

from oneopes.block_run import load_run, run_block
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import select_g1_g2, write_g1_g2_yaml
from oneopes.provenance import provenance
from oneopes.reports import write_hydration_files, write_run_metadata
from oneopes.system import SystemOptions, load_universe, select_system
from oneopes.trajectory_blocks import FrameWindow
from oneopes.waterfp import HydrationOptions, measure_hydration

logger = logging.getLogger(__name__)


def ligand_atoms(u: mda.Universe, hydration: xr.Dataset) -> AtomGroup:
    """The atoms of ``u`` that ``hydration`` was measured around, in its order (its ``atom_index``).

    Raises InputError unless they exist and carry the measured atom names, i.e. ``u`` has the
    topology the hydration was measured with.
    """
    indices = hydration.atom_index.values
    if indices.max() >= u.atoms.n_atoms or not np.array_equal(u.atoms[indices].names, hydration.atom.values):
        raise InputError("The topology's atoms do not match the measured ones: use the topology of the run.")
    return u.atoms[indices]


def write_selection(ligand: AtomGroup, hydration: xr.Dataset, outdir: Path, fp_round_decimals: int) -> None:
    """Write rdf.csv and fingerprints.csv of the ligand's ``hydration`` to ``outdir``, then its G1/G2
    selection as g1_g2.yaml; raises InputError when there is no selection (after writing the CSVs)."""
    write_hydration_files(hydration, outdir)
    selection = select_g1_g2(ligand, hydration.fp.values, fp_round_decimals)
    write_g1_g2_yaml(selection, outdir / "g1_g2.yaml")
    logger.info("%s; wrote rdf.csv, fingerprints.csv and g1_g2.yaml to %s", selection, outdir)


def select(*, tpr: ExistingFile, xtc: ExistingFile, outdir: Path,
           system: Annotated[SystemOptions, Parameter(name="*")] = SystemOptions(),
           window: Annotated[FrameWindow, Parameter(name="*")] = FrameWindow(),
           hydration: Annotated[HydrationOptions, Parameter(name="*")] = HydrationOptions(),
           fp_round_decimals: Annotated[int, Parameter(validator=Number(gte=0))] = 1) -> None:
    """Select G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on, from WaterFP over a frame window.

    The monitor already does this for its converged block; use this command for any other
    frame window. Writes, for review, to --outdir:
    rdf.csv (atom, r_nm, n_r: water-O number density, waters/nm^3),
    fingerprints.csv (atom, fp: WaterFP per ligand heavy atom) and
    g1_g2.yaml (G1 and G2 atom names and 1-based serials, MDAnalysis index + 1).

    Parameters
    ----------
    tpr
        Topology with bonds.
    xtc
        Trajectory the frame window is taken from.
    outdir
        Directory for rdf.csv, fingerprints.csv, g1_g2.yaml and run_metadata.json.
    fp_round_decimals
        Decimals FP is rounded to when comparing atoms.
    """
    u = load_universe(tpr, xtc)
    sel = select_system(u, system)
    ligand, water = sel.ligand(u), sel.water(u)

    frames = window.frames(len(u.trajectory))
    measurement = measure_hydration(ligand, water, frames, hydration)
    logger.info("%d ligand heavy atoms, frames %d-%d, FP from most negative:\n%s", len(ligand),
                frames.start, frames.stop - 1, measurement.fp.sortby(measurement.fp).to_series().to_string())
    options = {"system": asdict(system), "window": asdict(window), "hydration": asdict(hydration),
               "fp_round_decimals": fp_round_decimals}
    write_run_metadata(outdir, provenance({"tpr": tpr, "xtc": xtc}) | {
        "ligand_selection": sel.ligand_selection, "water_selection": sel.water_selection, "frames": frames,
        "options": options})
    write_selection(ligand, measurement, outdir, fp_round_decimals)


def select_block(*, tpr: ExistingFile, run: ExistingFile, outdir: Path, block: int | None = None,
                 fp_round_decimals: Annotated[int, Parameter(validator=Number(gte=0))] = 1) -> None:
    """Select G1/G2 from one block of a saved block run, by default its converged block.

    Takes the FP of the block from --run, the blocks.nc of `oneopes monitor`, so the trajectory is
    not read again. Writes rdf.csv, fingerprints.csv and g1_g2.yaml to --outdir, as `oneopes select`.

    Parameters
    ----------
    tpr
        Topology with bonds, the one the run was made with.
    run
        blocks.nc written by `oneopes monitor`.
    outdir
        Directory for rdf.csv, fingerprints.csv, g1_g2.yaml and run_metadata.json.
    block
        Block to select on (default: the converged block).
    fp_round_decimals
        Decimals FP is rounded to when comparing atoms.
    """
    hydration = run_block(load_run(run), block)
    write_run_metadata(outdir, provenance({"tpr": tpr, "run": run}) | {
        "block": int(hydration.block), "fp_round_decimals": fp_round_decimals})
    write_selection(ligand_atoms(load_universe(tpr), hydration), hydration, outdir, fp_round_decimals)
