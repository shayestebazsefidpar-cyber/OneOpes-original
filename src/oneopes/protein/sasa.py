"""
Static per-atom SASA (A^2) of the candidate atoms.

Computed on the complex structure's protein and ligand heavy atoms, without water or ions, with
freesasa's default algorithm (Lee & Richards) and radii.
"""

import numpy as np
import pandas as pd

from oneopes.protein.inputs import ProteinSystem


def static_sasa(system: ProteinSystem, candidate_indices: np.ndarray) -> np.ndarray:
    """SASA (A^2) of each candidate atom, in ``candidate_indices`` order, in the complex structure."""
    import freesasa

    freesasa.setVerbosity(freesasa.silent)
    atoms = system.selection.protein(system.complex) | system.selection.ligand(system.complex)
    x, y, z = atoms.positions.astype(float).T.tolist()
    structure = freesasa.Structure()
    structure.addAtoms(atoms.names.tolist(), atoms.resnames.tolist(), atoms.resids.astype(str).tolist(),
                       ["A"] * len(atoms), x, y, z)
    if structure.nAtoms() != len(atoms):
        raise RuntimeError(f"freesasa accepted {structure.nAtoms()} of {len(atoms)} atoms.")
    result = freesasa.calc(structure)
    area = pd.Series([result.atomArea(i) for i in range(len(atoms))], index=atoms.indices)
    return area.loc[candidate_indices].to_numpy()
