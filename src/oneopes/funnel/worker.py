"""Run by ``oneopes funnel`` in a Python with BioSimSpace: write the funnel CV of a complex as JSON.

Usage: ``python -m oneopes.funnel.worker TOPOLOGY COORDINATES STRUCTURE PROTEIN LIGAND OUT`` (STRUCTURE may be '').
"""

import json
import sys
from dataclasses import asdict
from pathlib import Path

from oneopes.funnel.bss import read_system
from oneopes.funnel.geometry import funnel_extent
from oneopes.funnel.make_funnel import make_funnel

EXTENT_POINTS_NM = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
"""Positions along the axis (nm) at which the funnel radius is written."""


def write_funnel(topology: Path, coordinates: Path, structure: Path | None, protein: int, ligand: int,
                 out: Path) -> None:
    """Build the funnel of molecules ``protein`` and ``ligand`` and write its atoms, axis and parameters to ``out``."""
    system = read_system(topology, coordinates, structure=structure)
    molecules = system.getMolecules()
    f = make_funnel(system, protein=protein, ligand=ligand)
    result = {
        "protein": {"molecule": protein, "atoms": molecules[protein].nAtoms()},
        "ligand": {"molecule": ligand, "atoms": molecules[ligand].nAtoms(),
                   "residue": molecules[ligand].getResidues()[0].name()},
        "atoms0_serial": [i + 1 for i in f.axis.atoms0],
        "atoms1_serial": [i + 1 for i in f.axis.atoms1],
        "p0_A": f.axis.p0.tolist(),
        "p1_A": f.axis.p1.tolist(),
        "direction": f.axis.direction.tolist(),
        "axis_length_A": f.axis.length,
        "parameters": asdict(f.parameters),
        "radius_nm": dict(zip(map(str, EXTENT_POINTS_NM), map(float, funnel_extent(f.cv, EXTENT_POINTS_NM)),
                              strict=True)),
    }
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    topology, coordinates, structure, protein, ligand, out = sys.argv[1:]
    write_funnel(Path(topology), Path(coordinates), Path(structure) if structure else None, int(protein),
                 int(ligand), Path(out))
