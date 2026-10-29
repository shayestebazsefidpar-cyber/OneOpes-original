"""G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on, chosen from WaterFP (``oneopes select``)."""
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import numpy as np
import yaml
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from MDAnalysis.core.groups import Atom, AtomGroup
from pydantic import BaseModel
from scipy.sparse.csgraph import shortest_path

from oneopes.system import SystemDefinitionError, select_system
from oneopes.waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP

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
    rules below refer to atoms by FP rank. Raises ValueError when a pair does not exist.
    """
    by_fp = np.argsort(fp, kind="stable")
    atoms, fp = atoms[by_fp], fp[by_fp]
    separation = bond_separation(atoms)
    scale = 10 ** fp_round_decimals
    g1 = anti_bulk_pair(np.rint(fp * scale).astype(int), round(SAME_FP_WITHIN * scale), separation)
    g2 = bulk_pair(separation)
    return SelectionResult(G1=tuple(map(AtomRef.of, atoms[list(g1)])), G2=tuple(map(AtomRef.of, atoms[list(g2)])))


def bond_separation(atoms: AtomGroup) -> np.ndarray:
    """Number of bonds between each pair of ``atoms`` (n x n), along bonds among them only.

    Raises ValueError unless the atoms form one bonded molecule (e.g. when bonds are missing).
    """
    position = {index: i for i, index in enumerate(atoms.indices)}
    bonded = np.zeros((len(atoms), len(atoms)), dtype=bool)
    for a, b in atoms.bonds.indices:
        if a in position and b in position:
            bonded[position[a], position[b]] = True
    separation = shortest_path(bonded, directed=False, unweighted=True)
    if np.isinf(separation).any():
        a, b = np.argwhere(np.isinf(separation))[0]
        raise ValueError(f"heavy atoms {atoms[a].name} and {atoms[b].name} are not connected by bonds")
    return separation.astype(int)


def bulk_pair(separation: np.ndarray) -> tuple[int, int]:
    """The last atom (least negative FP), and the latest one before it at least three bonds away."""
    anchor = len(separation) - 1
    partner = next((atom for atom in reversed(range(anchor)) if separation[anchor, atom] >= 3), None)
    if partner is None:
        raise ValueError("no heavy atom is three or more bonds from the least negative FP atom: no bulk partner")
    return anchor, partner


def anti_bulk_pair(fp_rounded: np.ndarray, same_fp_within: int, separation: np.ndarray) -> tuple[int, int]:
    """Atom 0 (most negative FP), and its partner among the atoms not bonded to it.

    ``fp_rounded`` are the rounded FPs as integers (-0.3 is -3 for one decimal), so that
    comparing them is exact; values less than ``same_fp_within`` apart count as the same FP.
    The partner starts as the first atom not bonded to atom 0. Walking on through atoms with
    the same FP as the partner, it moves to any atom farther from atom 0; the walk ends at the
    first atom with a different FP or exactly as far as the partner.
    """
    if np.count_nonzero(fp_rounded == fp_rounded[0]) > 2:
        raise ValueError("more than two atoms share the most negative rounded FP: the anti-bulk pair is ambiguous")
    not_bonded = [atom for atom in range(1, len(separation)) if separation[0, atom] > 1]
    if not not_bonded:
        raise ValueError("every other heavy atom is bonded to the most negative FP atom: no anti-bulk partner")
    partner, *later = not_bonded
    for atom in later:
        if abs(fp_rounded[atom] - fp_rounded[partner]) >= same_fp_within:
            break
        if separation[0, atom] == separation[0, partner]:
            break
        if separation[0, atom] > separation[0, partner]:
            partner = atom
    return 0, partner


def write_waterfp_files(waterfp: WaterFP, outdir: Path) -> None:
    """Write rdf.csv (atom, r_nm, n_r) and fingerprints.csv (atom, fp) of a finished WaterFP run."""
    outdir.mkdir(parents=True, exist_ok=True)
    waterfp.profile_table()[["atom", "r_nm", "n_r"]].to_csv(outdir / "rdf.csv", index=False)
    waterfp.fp_table().rename(columns={"FP": "fp"}).to_csv(outdir / "fingerprints.csv", index=False)


def write_g1_g2_yaml(selection: SelectionResult, path: Path) -> None:
    """Write the selection to YAML, creating the parent directory if needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(selection.model_dump(mode="json"), f, sort_keys=False)


def read_g1_g2_yaml(path: Path) -> SelectionResult:
    """The selection in a g1_g2.yaml; raises pydantic's ValidationError when the file is malformed."""
    return SelectionResult.model_validate(yaml.safe_load(Path(path).read_text()))


def select(*, tpr: ExistingFile, xtc: ExistingFile, outdir: Path,
           ligand_selection: str | None = None, water_selection: str | None = None,
           start_frame: int = 0, end_frame: int | None = None,
           grid: Annotated[RadialGrid, Parameter(name="*")] = RadialGrid(),
           norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT, fp_round_decimals: int = 1) -> None:
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
    ligand_selection
        MDAnalysis selection of the ligand (default: auto-detect).
    water_selection
        MDAnalysis selection of the water (default: known water residue names).
    end_frame
        Exclusive end frame (default: end of the trajectory).
    norm_tail_bins
        Bins at the end of the profile whose mean is the bulk density.
    fp_round_decimals
        Decimals FP is rounded to when comparing atoms.
    """
    u = mda.Universe(str(tpr), str(xtc))
    try:
        sel = select_system(u, ligand_selection, water_selection)
    except SystemDefinitionError as e:
        raise SystemExit(f"[select] {e}")
    ligand, water = u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens]

    waterfp = WaterFP(ligand, water, grid=grid, norm_tail_bins=norm_tail_bins)
    waterfp.run(start=start_frame, stop=end_frame)
    print(f"[select] {len(ligand)} ligand heavy atoms, frames {waterfp.frames[0]}-{waterfp.frames[-1]}")
    print(waterfp.fp_table().sort_values("FP", kind="stable").to_string(index=False))

    write_waterfp_files(waterfp, outdir)
    try:
        selection = select_g1_g2(ligand, waterfp.results.fp, fp_round_decimals)
    except ValueError as e:
        raise SystemExit(f"[select] no G1/G2 selection: {e}")
    write_g1_g2_yaml(selection, outdir / "g1_g2.yaml")
    print(f"[select] {selection}")
    print(f"[select] wrote rdf.csv, fingerprints.csv and g1_g2.yaml to {outdir}")
