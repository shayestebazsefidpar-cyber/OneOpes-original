"""Shared fixtures for comparisons against BioSimSpace (public API only).

Validation systems are only test inputs; nothing in ``src/`` depends on them.
They are located (read-only) from, in order:

1. ``ONEOPES_FUNNEL_REF_DIRS`` (os.pathsep-separated directories, each with
   ``top.top`` + ``npt*.gro`` + ``whole.pdb``), or
2. the OneOPES repository's ``original/`` tree, searched upwards from this
   file (``<repo>/original`` when the package lives in
   ``<repo>/workflows/oneopes_funnel``, or a sibling ``OneOpes-original``
   checkout during local development).

BioSimSpace-dependent tests skip when BioSimSpace or the systems are missing.
"""

import glob
import os
from dataclasses import dataclass
from pathlib import Path

import pytest

VALIDATION_SYSTEMS = ("HSP90/OneOPES/lig1", "BRD4/OneOPES/lig1")


def _find_original_tree():
    for parent in Path(__file__).resolve().parents:
        for candidate in (parent / "original", parent / "OneOpes-original" / "original"):
            if (candidate / VALIDATION_SYSTEMS[0]).is_dir():
                return candidate
    return None


def _reference_dirs():
    env = os.environ.get("ONEOPES_FUNNEL_REF_DIRS")
    if env:
        return [d for d in env.split(os.pathsep) if d]
    tree = _find_original_tree()
    return [str(tree / s) for s in VALIDATION_SYSTEMS] if tree else ["<no validation systems found>"]


REF_DIRS = _reference_dirs()


@dataclass
class RefSystem:
    """A validation system: BSS System (from top.top + npt*.gro) and its directory."""

    system: object
    directory: str


@pytest.fixture(scope="session", params=REF_DIRS, ids=lambda d: "/".join(d.rstrip("/").split("/")[-3:]))
def ref(request):
    BSS = pytest.importorskip("BioSimSpace")
    top = os.path.join(request.param, "top.top")
    gros = sorted(glob.glob(os.path.join(request.param, "npt*.gro")))
    if not (os.path.exists(top) and gros):
        pytest.skip(f"validation system (top.top + npt*.gro) not found in {request.param}")
    return RefSystem(system=BSS.IO.readMolecules([top, gros[0]]), directory=request.param)


def molecule_layout(system):
    """(protein_index, ligand_index, atom offsets, atom counts) by BSS's largest/second-largest rule."""
    counts = [m.nAtoms() for m in system]
    ranked = sorted((n, i) for i, n in enumerate(counts))
    offsets = [sum(counts[:i]) for i in range(len(counts))]
    return ranked[-1][1], ranked[-2][1], offsets, counts


def bss_xyz(atom):
    """BSS atom coordinates in Angstrom."""
    c = atom.coordinates()
    return [c.x().angstroms().value(), c.y().angstroms().value(), c.z().angstroms().value()]
