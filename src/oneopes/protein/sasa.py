"""
Static per-atom SASA (A^2) of the candidate atoms.

One structure only (the complex), protein + ligand heavy atoms, no water or
ions; freesasa's default algorithm (Lee & Richards) and default radii. Atoms
are added to freesasa in topology order, so every SASA value maps to its atom
by index - no file is written or reused.
"""

from __future__ import annotations

import numpy as np

from oneopes.system import heavy_mask


def static_sasa(u, protein_all, ligand_all, candidate_indices) -> np.ndarray:
    """SASA (A^2) of each candidate atom, in ``candidate_indices`` order, on ``u``'s current frame."""
    import freesasa

    freesasa.setVerbosity(freesasa.silent)
    atoms = u.atoms[np.union1d(np.asarray(protein_all), np.asarray(ligand_all))]
    atoms = atoms[heavy_mask(atoms)]
    structure = freesasa.Structure()
    for atom in atoms:
        x, y, z = (float(c) for c in atom.position)
        structure.addAtom(atom.name, atom.resname, str(atom.resid), "A", x, y, z)
    if structure.nAtoms() != len(atoms):
        raise RuntimeError(f"freesasa accepted {structure.nAtoms()} of {len(atoms)} atoms.")
    result = freesasa.calc(structure)
    position = {int(i): k for k, i in enumerate(atoms.indices)}
    missing = [int(i) for i in candidate_indices if int(i) not in position]
    if missing:
        raise RuntimeError(f"Candidate atoms missing from the SASA structure: {missing[:5]}")
    return np.array([result.atomArea(position[int(i)]) for i in candidate_indices])
