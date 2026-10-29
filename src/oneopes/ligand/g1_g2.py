"""G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on, chosen from WaterFP (``oneopes select``)."""
from collections.abc import Sequence
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
"""Rounded FP values closer than this count as the same FP."""


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
    Atoms are handled by their position in ``atoms``; raises ValueError when a pair does not exist.
    """
    by_fp = np.argsort(fp, kind="stable")
    separation = bond_separation(atoms)
    g1 = anti_bulk_pair(by_fp, np.round(fp, fp_round_decimals), separation)
    g2 = bulk_pair(by_fp, separation)
    return SelectionResult(G1=tuple(AtomRef.of(atoms[i]) for i in g1), G2=tuple(AtomRef.of(atoms[i]) for i in g2))


def bond_separation(atoms: AtomGroup) -> np.ndarray:
    """Number of bonds between each pair of ``atoms`` (n x n), along bonds among them only.

    Raises ValueError unless the atoms form one bonded molecule (e.g. a topology without bonds).
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


def bulk_pair(by_fp: Sequence[int], separation: np.ndarray) -> tuple[int, int]:
    """The least negative FP atom, and the next least negative one at least three bonds away."""
    anchor, *others = reversed(by_fp)
    partner = next((atom for atom in others if separation[anchor, atom] >= 3), None)
    if partner is None:
        raise ValueError("no heavy atom is three or more bonds from the least negative FP atom: no bulk partner")
    return anchor, partner


def anti_bulk_pair(by_fp: Sequence[int], fp_rounded: np.ndarray, separation: np.ndarray) -> tuple[int, int]:
    """The most negative FP atom, and a partner not bonded to it (see :func:`anti_bulk_partner`)."""
    anchor, *others = by_fp
    if np.count_nonzero(fp_rounded == fp_rounded[anchor]) > 2:
        raise ValueError("more than two atoms share the most negative rounded FP: the anti-bulk pair is ambiguous")
    not_bonded = [atom for atom in others if separation[anchor, atom] > 1]
    if not not_bonded:
        raise ValueError("every other heavy atom is bonded to the most negative FP atom: no anti-bulk partner")
    return anchor, anti_bulk_partner(not_bonded, fp_rounded, separation[anchor])


def anti_bulk_partner(candidates: Sequence[int], fp_rounded: np.ndarray, bonds_to_anchor: np.ndarray) -> int:
    """The upstream notebook's partner rule, applied to the candidates in FP order.

    The partner starts as the first candidate. A later candidate with the same rounded FP as the
    current partner (within SAME_FP_WITHIN) replaces it when it is farther from the anchor; one
    exactly as far ends the search. Candidates with a different FP are passed over.
    """
    partner, *later = candidates
    for atom in later:
        if abs(fp_rounded[atom] - fp_rounded[partner]) >= SAME_FP_WITHIN:
            continue
        if bonds_to_anchor[atom] > bonds_to_anchor[partner]:
            partner = atom
        elif bonds_to_anchor[atom] == bonds_to_anchor[partner]:
            break
    return partner


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
