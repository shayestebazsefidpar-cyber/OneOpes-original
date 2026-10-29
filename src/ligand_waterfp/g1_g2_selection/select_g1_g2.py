"""
G1/G2 selection: the ligand heavy-atom pairs the ligand-side CV is built on, chosen from WaterFP.

    ligand-waterfp-select --tpr prod.tpr --xtc prod.xtc --outdir outputs/selection \\
        [--start-frame N] [--end-frame M] [--ligand-selection "resname MOL"]

The monitor already does this for its converged block; use this command for any other frame
window. Writes, for review:
    rdf.csv           (atom,r_nm,n_r)  water-O number density, waters/nm^3
    fingerprints.csv  (atom,fp)        WaterFP per ligand heavy atom
    g1_g2.yaml        G1 and G2 atom names and 1-based serials (MDAnalysis index + 1)
"""
import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import MDAnalysis as mda
import networkx as nx
import numpy as np
import yaml
from MDAnalysis.core.groups import AtomGroup

from system import SystemDefinitionError, select_system
from waterfp import NORM_TAIL_BINS_DEFAULT, RadialGrid, WaterFP

SAME_FP = 0.2  # rounded FP values closer than this count as the same FP


@dataclass(frozen=True)
class AtomRef:
    """An atom's name and 1-based serial."""
    name: str
    serial: int

    @classmethod
    def from_dict(cls, d: dict) -> "AtomRef":
        return cls(name=d["name"], serial=int(d["serial"]))

    def to_dict(self) -> dict:
        return {"name": self.name, "serial": self.serial}


@dataclass(frozen=True)
class SelectionResult:
    """The G1 (anti-bulk) and G2 (bulk) atom pairs."""
    G1: tuple[AtomRef, AtomRef]
    G2: tuple[AtomRef, AtomRef]

    @classmethod
    def from_dict(cls, d: dict) -> "SelectionResult":
        return cls(G1=tuple(AtomRef.from_dict(a) for a in d["G1"]),
                   G2=tuple(AtomRef.from_dict(a) for a in d["G2"]))

    def to_yaml_dict(self) -> dict:
        return {"G1": [a.to_dict() for a in self.G1], "G2": [a.to_dict() for a in self.G2]}

    def __str__(self) -> str:
        return (f"G1 (anti-bulk): {self.G1[0].name}, {self.G1[1].name}; "
                f"G2 (bulk): {self.G2[0].name}, {self.G2[1].name}")


def write_g1_g2_yaml(selection: SelectionResult, out_path: Path) -> None:
    """Write the selection to YAML, creating the parent directory if needed."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        yaml.safe_dump(selection.to_yaml_dict(), f, sort_keys=False)


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


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tpr", required=True, help="topology with bonds")
    p.add_argument("--xtc", required=True)
    p.add_argument("--outdir", required=True)
    p.add_argument("--ligand-selection", default=None, help="MDAnalysis selection (default: auto-detect)")
    p.add_argument("--water-selection", default=None, help="MDAnalysis selection (default: known water names)")
    p.add_argument("--start-frame", type=int, default=0)
    p.add_argument("--end-frame", type=int, default=None)
    p.add_argument("--rmax-nm", type=float, default=RadialGrid.rmax_nm)
    p.add_argument("--binwidth-nm", type=float, default=RadialGrid.binwidth_nm)
    p.add_argument("--norm-tail-bins", type=int, default=NORM_TAIL_BINS_DEFAULT)
    p.add_argument("--fp-round-decimals", type=int, default=1, help="decimals FP is rounded to when comparing atoms")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    outdir = Path(args.outdir)

    u = mda.Universe(args.tpr, args.xtc)
    try:
        sel = select_system(u, args.ligand_selection, args.water_selection)
    except SystemDefinitionError as e:
        raise SystemExit(f"[select] {e}")
    ligand, water = u.atoms[sel.ligand_heavy], u.atoms[sel.water_oxygens]

    grid = RadialGrid(rmax_nm=args.rmax_nm, binwidth_nm=args.binwidth_nm)
    waterfp = WaterFP(ligand, water, grid=grid, norm_tail_bins=args.norm_tail_bins)
    waterfp.run(start=args.start_frame, stop=args.end_frame)
    print(f"[select] {len(ligand)} ligand heavy atoms, frames {waterfp.frames[0]}-{waterfp.frames[-1]}")
    print(waterfp.fp_table().sort_values("FP", kind="stable").to_string(index=False))

    try:
        selection = write_selection(waterfp, outdir, args.fp_round_decimals)
    except ValueError as e:
        raise SystemExit(f"[select] no G1/G2 selection: {e}")
    print(f"[select] {selection}")
    print(f"[select] wrote rdf.csv, fingerprints.csv and g1_g2.yaml to {outdir}")


if __name__ == "__main__":
    main()
