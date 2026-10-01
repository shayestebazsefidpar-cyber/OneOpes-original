"""Tests for oneopes_funnel.p0_p1: BSS makeFunnel() selections + derived P0/P1."""

import numpy as np
import pytest
from conftest import bss_xyz, molecule_layout

from oneopes_funnel.p0_p1 import centroid, p0_p1_from_atoms


# --------------------------------------------------------------------------
# B: unit tests (no BioSimSpace)
# --------------------------------------------------------------------------
def test_b2_centroid_is_plain_mean():
    coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 4.0, 0.0], [2.0, 4.0, 6.0]])
    np.testing.assert_allclose(centroid(coords), [1.0, 2.0, 1.5], atol=1e-12)
    with pytest.raises(ValueError):
        centroid(np.empty((0, 3)))


def test_b3_p0_p1_from_system_level_indices():
    # A 3-atom "water" first, so system indices != protein-local indices.
    coords = np.array(
        [
            [50.0, 50.0, 50.0], [50.8, 50.0, 50.0], [50.0, 50.8, 50.0],  # water (0-2)
            [0.0, 0.0, 0.0], [2.0, 0.0, 0.0],                             # atoms0 (3, 4)
            [9.0, 9.0, 9.0],                                              # unrelated (5)
            [0.0, 0.0, 8.0], [2.0, 0.0, 8.0],                             # atoms1 (6, 7)
        ]
    )
    r = p0_p1_from_atoms([3, 4], [6, 7], coords)
    assert (r.atoms0, r.atoms1) == ([3, 4], [6, 7])
    np.testing.assert_allclose(r.p0, [1.0, 0.0, 0.0], atol=1e-12)
    np.testing.assert_allclose(r.p1, [1.0, 0.0, 8.0], atol=1e-12)
    np.testing.assert_allclose(r.axis, [0.0, 0.0, 1.0], atol=1e-12)
    assert r.axis_length == pytest.approx(8.0, abs=1e-12)


# --------------------------------------------------------------------------
# C: direct comparison with BSS makeFunnel() on the validation systems
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def bss_funnel_atoms(ref):
    from BioSimSpace.Metadynamics.CollectiveVariable import makeFunnel

    return makeFunnel(ref.system)


@pytest.fixture(scope="module")
def ours(ref):
    from oneopes_funnel.p0_p1 import make_p0_p1

    return make_p0_p1(ref.system)


def test_c1_c2_atoms_equal_bss_makefunnel(ours, bss_funnel_atoms):
    assert ours.atoms0 == list(bss_funnel_atoms[0])
    assert ours.atoms1 == list(bss_funnel_atoms[1])


def test_c3_arguments_pass_through_to_makefunnel(ref):
    from BioSimSpace.Metadynamics.CollectiveVariable import makeFunnel

    from oneopes_funnel.p0_p1 import make_p0_p1

    system = ref.system
    protein, ligand, _, _ = molecule_layout(system)
    for kwargs in (
        dict(protein=protein, ligand=ligand),
        dict(protein=system[protein], ligand=system[ligand]),
        dict(alpha_carbon_name="CA", property_map={}),
    ):
        expected0, expected1 = makeFunnel(system, **kwargs)
        r = make_p0_p1(system, **kwargs)
        assert (r.atoms0, r.atoms1) == (list(expected0), list(expected1)), kwargs


def test_c4_c5_p0_p1_derived_from_bss_selected_atoms(ref, ours, bss_funnel_atoms):
    # makeFunnel returns atom selections only; P0/P1 are DERIVED as the
    # average BSS coordinate (A) of exactly those atoms.
    atoms0, atoms1 = bss_funnel_atoms
    expected_p0 = np.mean([bss_xyz(ref.system.getAtom(i)) for i in atoms0], axis=0)
    expected_p1 = np.mean([bss_xyz(ref.system.getAtom(i)) for i in atoms1], axis=0)
    np.testing.assert_allclose(ours.p0, expected_p0, atol=1e-9)
    np.testing.assert_allclose(ours.p1, expected_p1, atol=1e-9)


def test_c6_axis_consistent_with_p0_p1(ours):
    np.testing.assert_allclose(ours.axis_length, np.linalg.norm(ours.p1 - ours.p0), atol=1e-12)
    np.testing.assert_allclose(ours.axis, (ours.p1 - ours.p0) / ours.axis_length, atol=1e-12)
    assert np.linalg.norm(ours.axis) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# D: atom-index conventions
# --------------------------------------------------------------------------
def test_d1_indices_are_bss_system_indices_of_ca_atoms(ref, ours):
    system = ref.system
    for i in ours.atoms0 + ours.atoms1:
        atom = system.getAtom(i)
        assert system.getIndex(atom) == i
        assert atom.name() == "CA"


def test_d2_indices_are_system_level_not_protein_local(ref, ours):
    protein, _, offsets, counts = molecule_layout(ref.system)
    protein_atoms = ref.system[protein].getAtoms()
    for i in ours.atoms0 + ours.atoms1:
        assert offsets[protein] <= i < offsets[protein] + counts[protein]
        # the same atom, looked up protein-locally, is at i - offset
        assert bss_xyz(protein_atoms[i - offsets[protein]]) == bss_xyz(ref.system.getAtom(i))


def test_d4_bss_index_plus_one_is_whole_pdb_serial(ref, ours):
    mda = pytest.importorskip("MDAnalysis")
    import os

    pdb = os.path.join(ref.directory, "whole.pdb")
    if not os.path.exists(pdb):
        pytest.skip(f"no whole.pdb in {ref.directory}")
    u = mda.Universe(pdb)
    # residue name of each protein atom, from public Residue.name()/nAtoms()
    protein, _, offsets, _ = molecule_layout(ref.system)
    residue_of = []
    for residue in ref.system[protein].getResidues():
        residue_of += [residue.name()] * residue.nAtoms()
    for i in ours.atoms0 + ours.atoms1:
        pdb_atom = u.atoms[i]  # 0-based position = PDB serial i + 1
        assert int(pdb_atom.id) == i + 1
        assert (pdb_atom.name, pdb_atom.resname) == (ref.system.getAtom(i).name(), residue_of[i - offsets[protein]])
