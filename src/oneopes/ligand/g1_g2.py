"""G1/G2, the ligand heavy-atom pairs the ligand-side CV is built on: the selection rule from WaterFP, and
its g1_g2.yaml file."""
from pathlib import Path

import numpy as np
import yaml
from MDAnalysis.core.groups import Atom, AtomGroup
from pydantic import BaseModel
from scipy.sparse import coo_array
from scipy.sparse.csgraph import connected_components, shortest_path

from oneopes.errors import InputError

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
    """G1 (anti-bulk) and G2 (bulk) pairs from the FP of the ligand's heavy atoms.

    The rules of WaterFP's fp_driven_atom_selection.ipynb; raises InputError when a pair does not exist.
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
    """Bonds between each atom at the positions ``sources`` and every atom of ``atoms``; one row per source.

    Raises InputError unless the bonds join ``atoms`` into one molecule.
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
    """The G1 partner of atom 0 (most negative FP): the first atom not bonded to it, or a farther one of the same FP.

    ``fp_rounded`` are integers (-0.3 is -3 for one decimal), so comparing them is exact.
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


def write_g1_g2_yaml(selection: SelectionResult, path: Path) -> None:
    """Write the selection to YAML, creating the parent directory if needed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(selection.model_dump(mode="json"), sort_keys=False))


def read_g1_g2_yaml(path: Path) -> SelectionResult:
    """The selection in a g1_g2.yaml; raises pydantic's ValidationError when the file is malformed."""
    return SelectionResult.model_validate(yaml.safe_load(path.read_text()))
