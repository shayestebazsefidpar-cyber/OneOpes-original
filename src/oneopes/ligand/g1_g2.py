"""G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on, chosen from WaterFP (``oneopes select``)."""
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Annotated

import MDAnalysis as mda
import networkx as nx
import numpy as np
import yaml
from cyclopts import Parameter
from cyclopts.types import ExistingFile
from MDAnalysis.core.groups import AtomGroup
from pydantic import BaseModel

from oneopes.system import SystemDefinitionError, select_system
from oneopes.waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP

SAME_FP = 0.2  # rounded FP values closer than this count as the same FP


class AtomRef(BaseModel, frozen=True):
    """An atom's name and 1-based serial."""
    name: str
    serial: int


class SelectionResult(BaseModel, frozen=True):
    """The G1 (anti-bulk) and G2 (bulk) atom pairs, as written to and read from g1_g2.yaml."""
    G1: tuple[AtomRef, AtomRef]
    G2: tuple[AtomRef, AtomRef]

    def __str__(self) -> str:
        return (f"G1 (anti-bulk): {self.G1[0].name}, {self.G1[1].name}; "
                f"G2 (bulk): {self.G2[0].name}, {self.G2[1].name}")


def write_g1_g2_yaml(selection: SelectionResult, out_path: Path) -> None:
    """Write the selection to YAML, creating the parent directory if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(selection.model_dump(mode="json"), f, sort_keys=False)


def write_selection(waterfp: WaterFP, outdir: Path, fp_round_decimals: int = 1) -> SelectionResult:
    """Write rdf.csv, fingerprints.csv and g1_g2.yaml for a finished WaterFP run, and return the selection.

    The CSVs are written even when no selection exists (ValueError).
    """
    outdir.mkdir(parents=True, exist_ok=True)
    waterfp.profile_table()[["atom", "r_nm", "n_r"]].to_csv(outdir / "rdf.csv", index=False)
    waterfp.fp_table().rename(columns={"FP": "fp"}).to_csv(outdir / "fingerprints.csv", index=False)
    selection = select_g1_g2(waterfp.solute, waterfp.results.fp, fp_round_decimals)
    write_g1_g2_yaml(selection, outdir / "g1_g2.yaml")
    return selection


def select_g1_g2(atoms: AtomGroup, fp: np.ndarray, fp_round_decimals: int = 1) -> SelectionResult:
    """G1 (anti-bulk pair) and G2 (bulk pair) among a ligand's heavy atoms, from their FP values.

    Same rules as WaterFP's fp_driven_atom_selection.ipynb (checked against it in the tests).
    Ties in FP keep the atoms' order.
    """
    order = list(np.argsort(fp, kind="stable"))  # positions in ``atoms``, most negative FP first
    distance = bond_distances(atoms)
    g1 = anti_bulk_pair(order, np.round(fp, fp_round_decimals), distance)
    g2 = bulk_pair(order, distance)

    def ref(position: int) -> AtomRef:
        return AtomRef(name=atoms[position].name, serial=int(atoms[position].index) + 1)

    return SelectionResult(G1=(ref(g1[0]), ref(g1[1])), G2=(ref(g2[0]), ref(g2[1])))


def anti_bulk_pair(order: Sequence[int], fp_round: np.ndarray, distance: Callable[[int, int], int]) -> tuple[int, int]:
    """The most negative FP atom, and the atom farthest from it among the next most negative FPs.

    Atoms bonded to the anchor are skipped. The partner is the first remaining atom, replaced by a
    later one with the same rounded FP (within SAME_FP) that is farther from the anchor; an equally
    far one stops the search.
    """
    anchor = order[0]
    if np.count_nonzero(fp_round == fp_round[anchor]) > 2:
        raise ValueError("more than two atoms share the most negative rounded FP: the anti-bulk pair is ambiguous")
    partner = None
    for candidate in order[1:]:
        d = distance(candidate, anchor)
        if d <= 1:
            continue  # bonded to the anchor
        if partner is None:
            partner = candidate
        elif abs(fp_round[candidate] - fp_round[partner]) < SAME_FP:
            if d > distance(partner, anchor):
                partner = candidate
            elif d == distance(partner, anchor):
                break
    if partner is None:
        raise ValueError("every other heavy atom is bonded to the most negative FP atom: no anti-bulk partner")
    return anchor, partner


def bulk_pair(order: Sequence[int], distance: Callable[[int, int], int]) -> tuple[int, int]:
    """The least negative FP atom, and the next least negative one at least three bonds away."""
    anchor = order[-1]
    partner = next((c for c in reversed(order[:-1]) if distance(c, anchor) > 2), None)
    if partner is None:
        raise ValueError("no heavy atom is three or more bonds from the least negative FP atom: no bulk partner")
    return anchor, partner


def bond_distances(atoms: AtomGroup) -> Callable[[int, int], int]:
    """Number of bonds between two of ``atoms`` (by position), counting only bonds among them."""
    position = {index: i for i, index in enumerate(atoms.indices)}
    graph = nx.Graph()
    graph.add_nodes_from(range(len(atoms)))
    graph.add_edges_from((position[a], position[b]) for a, b in atoms.bonds.indices
                         if a in position and b in position)
    lengths = dict(nx.all_pairs_shortest_path_length(graph))

    def distance(a: int, b: int) -> int:
        if b not in lengths[a]:
            raise ValueError(f"heavy atoms {atoms[a].name} and {atoms[b].name} are not connected by bonds")
        return lengths[a][b]

    return distance


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

    try:
        selection = write_selection(waterfp, outdir, fp_round_decimals)
    except ValueError as e:
        raise SystemExit(f"[select] no G1/G2 selection: {e}")
    print(f"[select] {selection}")
    print(f"[select] wrote rdf.csv, fingerprints.csv and g1_g2.yaml to {outdir}")
