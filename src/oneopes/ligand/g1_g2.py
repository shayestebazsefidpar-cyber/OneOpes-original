"""G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on, chosen from WaterFP (``oneopes select``)."""
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import numpy as np
import xarray as xr
import yaml
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from cyclopts.validators import Number
from MDAnalysis.core.groups import Atom, AtomGroup
from pydantic import BaseModel
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components, shortest_path

from oneopes.block_run import load_run, run_block
from oneopes.errors import InputError
from oneopes.provenance import provenance
from oneopes.reports import write_hydration_files, write_run_metadata
from oneopes.system import SystemOptions, load_universe, select_system
from oneopes.trajectory_blocks import FrameWindow
from oneopes.waterfp import HydrationOptions, measure_hydration

logger = logging.getLogger(__name__)

SAME_FP_WITHIN = 0.2
"""Rounded FP values less than this apart count as the same FP."""


class AtomRef(BaseModel, frozen=True):
    """An atom's name and 1-based serial."""
    name: str
    serial: int

    @classmethod
    def of(cls, atom: Atom) -> "AtomRef":
        return cls(name=atom.name, serial=int(atom.index) + 1)


class SelectionResult(BaseModel, frozen=True):
    """The G1 (anti-bulk) and G2 (bulk) atom pairs, as written to and read from g1_g2.yaml."""
    G1: tuple[AtomRef, AtomRef]
    G2: tuple[AtomRef, AtomRef]

    def __str__(self) -> str:
        return (f"G1 (anti-bulk): {self.G1[0].name}, {self.G1[1].name}; "
                f"G2 (bulk): {self.G2[0].name}, {self.G2[1].name}")


def select_g1_g2(atoms: AtomGroup, fp: np.ndarray, fp_round_decimals: int = 1) -> SelectionResult:
    """G1 (anti-bulk pair) and G2 (bulk pair) among a ligand's heavy atoms, from their FP values.

    The rules of WaterFP's fp_driven_atom_selection.ipynb, checked against it in the tests.
    The atoms are put in FP order first (most negative first, ties keeping their order), so the
    rules below refer to atoms by FP rank. Raises InputError when a pair does not exist.
    """
    by_fp = np.argsort(fp, kind="stable")
    atoms, fp = atoms[by_fp], fp[by_fp]
    first, last = 0, len(atoms) - 1
    bonds_from_first, bonds_from_last = bonds_from(atoms, [first, last])
    scale = 10 ** fp_round_decimals
    g1 = anti_bulk_partner(np.rint(fp * scale).astype(int), round(SAME_FP_WITHIN * scale), bonds_from_first)
    g2 = bulk_partner(bonds_from_last)
    return SelectionResult(G1=(AtomRef.of(atoms[first]), AtomRef.of(atoms[g1])),
                           G2=(AtomRef.of(atoms[last]), AtomRef.of(atoms[g2])))


def bonds_from(atoms: AtomGroup, sources: list[int]) -> np.ndarray:
    """Number of bonds from each atom at the positions ``sources`` to every atom of ``atoms``, along the
    bonds among ``atoms`` only; one row per source.

    Raises InputError unless those bonds join the atoms into one molecule (e.g. when bonds are missing).
    """
    bonds = atoms.intra_bonds.indices
    n_topology = atoms.universe.atoms.n_atoms
    bond_graph = coo_array((np.ones(len(bonds)), bonds.T), shape=(n_topology, n_topology)).tocsr()
    bond_graph = bond_graph[atoms.indices][:, atoms.indices]

    n_pieces, piece_of = connected_components(bond_graph, directed=False)
    if n_pieces > 1:
        pieces = [", ".join(atoms[piece_of == piece].names) for piece in range(n_pieces)]
        raise InputError(f"the heavy atoms are not one bonded molecule but {n_pieces}: {' | '.join(pieces)}")
    return shortest_path(bond_graph, directed=False, unweighted=True, indices=sources).astype(int)


def bulk_partner(bonds_from_last: np.ndarray) -> int:
    """The G2 partner of the last atom (least negative FP): the latest atom before it at least three bonds away."""
    far = np.flatnonzero(bonds_from_last[:-1] >= 3)
    if not far.size:
        raise InputError("no heavy atom is three or more bonds from the least negative FP atom: no bulk partner")
    return int(far[-1])


def anti_bulk_partner(fp_rounded: np.ndarray, same_fp_within: int, bonds_from_first: np.ndarray) -> int:
    """The G1 partner of atom 0 (most negative FP), among the atoms not bonded to it.

    ``fp_rounded`` are the rounded FPs as integers (-0.3 is -3 for one decimal), so that
    comparing them is exact; values less than ``same_fp_within`` apart count as the same FP.
    The partner starts as the first atom not bonded to atom 0. Walking on through atoms with
    the same FP as the partner, it moves to any atom farther from atom 0; the walk ends at the
    first atom with a different FP or exactly as far as the partner.
    """
    if np.count_nonzero(fp_rounded == fp_rounded[0]) > 2:
        raise InputError("more than two atoms share the most negative rounded FP: the anti-bulk pair is ambiguous")
    not_bonded = np.flatnonzero(bonds_from_first > 1)
    if not not_bonded.size:
        raise InputError("every other heavy atom is bonded to the most negative FP atom: no anti-bulk partner")
    partner, *later = not_bonded
    for atom in later:
        if abs(fp_rounded[atom] - fp_rounded[partner]) >= same_fp_within:
            break
        if bonds_from_first[atom] == bonds_from_first[partner]:
            break
        if bonds_from_first[atom] > bonds_from_first[partner]:
            partner = atom
    return int(partner)


def ligand_atoms(u: mda.Universe, hydration: xr.Dataset) -> AtomGroup:
    """The atoms of ``u`` that ``hydration`` was measured around, in its order (its ``atom_index``).

    Raises InputError unless they exist and carry the measured atom names, i.e. ``u`` has the
    topology the hydration was measured with.
    """
    indices = hydration.atom_index.values
    if indices.max() >= u.atoms.n_atoms or not np.array_equal(u.atoms[indices].names, hydration.atom.values):
        raise InputError("The topology's atoms do not match the measured ones: use the topology of the run.")
    return u.atoms[indices]


def write_selection(ligand: AtomGroup, hydration: xr.Dataset, outdir: Path, fp_round_decimals: int) -> SelectionResult:
    """Write rdf.csv and fingerprints.csv of the ligand's ``hydration`` to ``outdir``, then its G1/G2
    selection as g1_g2.yaml; raises InputError when there is no selection (after writing the CSVs)."""
    write_hydration_files(hydration, outdir)
    selection = select_g1_g2(ligand, hydration.fp.values, fp_round_decimals)
    write_g1_g2_yaml(selection, outdir / "g1_g2.yaml")
    logger.info("%s; wrote rdf.csv, fingerprints.csv and g1_g2.yaml to %s", selection, outdir)
    return selection


def write_g1_g2_yaml(selection: SelectionResult, path: Path) -> None:
    """Write the selection to YAML, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(selection.model_dump(mode="json"), sort_keys=False))


def read_g1_g2_yaml(path: Path) -> SelectionResult:
    """The selection in a g1_g2.yaml; raises pydantic's ValidationError when the file is malformed."""
    return SelectionResult.model_validate(yaml.safe_load(path.read_text()))


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
    fp_round_decimals
        Decimals FP is rounded to when comparing atoms.
    """
    u = load_universe(tpr, xtc)
    sel = select_system(u, system)
    ligand, water = u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens]

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
    block
        Block to select on (default: the converged block).
    fp_round_decimals
        Decimals FP is rounded to when comparing atoms.
    """
    hydration = run_block(load_run(run), block)
    write_run_metadata(outdir, provenance({"tpr": tpr, "run": run}) | {
        "block": int(hydration.block), "fp_round_decimals": fp_round_decimals})
    write_selection(ligand_atoms(load_universe(tpr), hydration), hydration, outdir, fp_round_decimals)
