"""The clean G1/G2 selection must agree with the vendored WaterFP notebook code on random ligands.

One known difference is removed before comparing: the notebook tests ``abs(fp_a - fp_b) < 0.2``
on rounded FPs in floating point, so values exactly 0.2 apart sometimes count as the same FP
(-0.1 - -0.3 = 0.19999999999999998) and sometimes not. Our selection compares them exactly.
The notebook reads each rounded FP through ``float()``; the test replaces that ``float`` with
one that maps every 0.1 step to 0.125, a binary-exact unit, so the notebook's comparison is
exact too ("less than 0.2 apart" = at most one 0.1 step).
"""
import builtins
import re
from types import SimpleNamespace

import MDAnalysis as mda
import numpy as np
import pandas as pd
import pytest

from oneopes.ligand.g1_g2 import AtomRef, G1G2Selection, select_g1_g2
from reference import vendored_selection as official

MOL = "SystemA-lig1"  # the vendored code's "<system>-<ligand>" id
SENTENCE = re.compile(r"selection: (\w+) \((\d+)\), (\w+) \((\d+)\)")  # "...selection: C3 (5), C9 (12)"


def random_ligand(rng: np.random.Generator) -> mda.Universe:
    """A ligand (resname MOL) whose heavy atoms form a random connected graph; some carry a hydrogen."""
    n_heavy = int(rng.integers(4, 15))
    heavy_bonds = [(i, int(rng.integers(0, i))) for i in range(1, n_heavy)]  # a random tree
    heavy_bonds += [tuple(sorted(rng.choice(n_heavy, 2, replace=False))) for _ in range(int(rng.integers(0, 3)))]

    names, heavy_index, bonds = [], {}, []
    for i in range(n_heavy):  # hydrogens interleaved, so heavy-atom positions differ from atom serials
        heavy_index[i] = len(names)
        names.append(f"C{i}")
        if rng.random() < 0.5:
            bonds.append((heavy_index[i], len(names)))
            names.append(f"H{i}")
    bonds += [(heavy_index[a], heavy_index[b]) for a, b in set(heavy_bonds) if a != b]

    u = mda.Universe.empty(len(names), n_residues=1, atom_resindex=[0] * len(names))
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", ["MOL"])
    u.add_TopologyAttr("bonds", bonds)
    return u


def pair(sentence: str) -> tuple[AtomRef, AtomRef]:
    name1, serial1, name2, serial2 = SENTENCE.search(sentence).groups()
    return AtomRef(name=name1, serial=int(serial1)), AtomRef(name=name2, serial=int(serial2))


def vendored_selection(u: mda.Universe, fp: np.ndarray, monkeypatch) -> G1G2Selection | None:
    """Run the vendored notebook functions; None when they report ambiguity or fail."""
    monkeypatch.setattr(official, "MDAnalysis", SimpleNamespace(Universe=lambda path: u))
    ranking = pd.DataFrame({"name": MOL, "atom": np.arange(1, len(fp) + 1), "fp": fp, "fp_round": np.round(fp, 1)})
    monkeypatch.setattr(official, "c", ranking, raising=False)
    monkeypatch.setattr(official, "float", lambda x: round(builtins.float(x) * 10) * 0.125, raising=False)
    try:
        anti_bulk = official.select_next_atom(MOL, verbose=False)
        bulk = official.select_bulk_atom(MOL, verbose=False)
    except Exception:  # the vendored code crashes when no partner exists
        return None
    if anti_bulk is None:  # ambiguous anchor
        return None
    return G1G2Selection(G1=pair(anti_bulk), G2=pair(bulk))


def test_matches_the_vendored_notebook_code_on_random_ligands(monkeypatch):
    rng = np.random.default_rng(0)
    agreed, both_rejected = 0, 0
    for _ in range(400):
        u = random_ligand(rng)
        heavy = u.select_atoms("resname MOL and not name H*")
        fp = -rng.uniform(0.0, 1.5, len(heavy))  # clustered: rounded-FP ties are common
        expected = vendored_selection(u, fp, monkeypatch)
        if expected is None:
            with pytest.raises(ValueError):
                select_g1_g2(heavy, fp)
            both_rejected += 1
        else:
            assert select_g1_g2(heavy, fp) == expected
            agreed += 1
    assert agreed > 100 and both_rejected > 10  # both outcomes are really exercised
