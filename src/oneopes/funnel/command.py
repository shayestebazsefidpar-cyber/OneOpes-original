"""The funnel CV of a protein-ligand complex (``oneopes funnel``), built in a Python with BioSimSpace."""

import json
import logging
import os
import subprocess
from datetime import datetime
from pathlib import Path

from cyclopts.types import ExistingFile

import oneopes
from oneopes.errors import InputError
from oneopes.provenance import provenance
from oneopes.reports import write_run_metadata

logger = logging.getLogger("oneopes")


def funnel(*, topology: ExistingFile, coordinates: ExistingFile, structure: ExistingFile | None = None,
           protein: int = 0, ligand: int = 1, outdir: Path | None = None, python: str = "python") -> None:
    """Build the funnel CV of a protein-ligand complex with BioSimSpace's makeFunnel().

    BioSimSpace is conda-only, so --python, a Python with BioSimSpace (e.g. the 'biosimspace' conda env),
    builds the funnel. Writes funnel.json (atoms0/atoms1 as GROMACS serials, P0, P1, axis and parameters),
    funnel_summary.txt and run_metadata.json to --outdir.

    Parameters
    ----------
    topology
        GROMACS topology of the complex (.top), for the molecule boundaries.
    coordinates
        Coordinates matching the topology (.gro), e.g. npt.gro.
    structure
        The same frame with whole molecules, e.g. from `gmx trjconv -pbc mol -o whole.pdb`; its positions
        replace those of --coordinates.
    protein
        Index of the protein molecule in the topology's [ molecules ] (0-based).
    ligand
        Index of the ligand molecule (0-based). makeFunnel() would otherwise take the second-largest
        molecule, which in a membrane system is a lipid.
    outdir
        Directory to write, which must not exist (default: funnel_YYYYMMDD_HHMMSS).
    python
        Python executable with BioSimSpace.
    """
    outdir = outdir or Path(f"funnel_{datetime.now():%Y%m%d_%H%M%S}")
    if outdir.exists():
        raise InputError(f"{outdir} exists; give a new --outdir.")
    outdir.mkdir(parents=True)
    out = outdir / "funnel.json"
    env = {**os.environ, "PYTHONPATH": str(Path(oneopes.__file__).parents[1])}
    command = [python, "-m", "oneopes.funnel.worker", str(topology), str(coordinates), str(structure or ""),
               str(protein), str(ligand), str(out)]
    try:
        result = subprocess.run(command, env=env, capture_output=True, text=True)
    except FileNotFoundError:
        raise InputError(f"Python not found as {python!r}; pass one with BioSimSpace with --python.") from None
    if not out.exists():
        output = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
        raise InputError(f"the funnel was not built; the end of the output of {python}:\n{output}")
    inputs = {"topology": topology, "coordinates": coordinates} | ({"structure": structure} if structure else {})
    write_run_metadata(outdir, provenance(inputs) | {
        "options": {"protein": protein, "ligand": ligand, "python": python}})
    summary = funnel_summary(json.loads(out.read_text()))
    (outdir / "funnel_summary.txt").write_text(summary)
    print(summary)
    logger.info("wrote %s, funnel_summary.txt and run_metadata.json", out)


def funnel_summary(f: dict) -> str:
    """The funnel of funnel.json as text."""
    p = f["parameters"]
    return "\n".join([
        f"protein: molecule {f['protein']['molecule']}, {f['protein']['atoms']} atoms",
        f"ligand: molecule {f['ligand']['molecule']} ({f['ligand']['residue']}), {f['ligand']['atoms']} atoms",
        f"atoms0 (P0, origin) serials: {f['atoms0_serial']}",
        f"atoms1 (P1, direction) serials: {f['atoms1_serial']}",
        f"P0 (A): {[round(x, 3) for x in f['p0_A']]}",
        f"P1 (A): {[round(x, 3) for x in f['p1_A']]}",
        f"axis length (A): {f['axis_length_A']:.2f}",
        f"width {p['width_nm']} nm, buffer {p['buffer_nm']} nm, steepness {p['steepness_per_nm']} /nm, "
        f"inflection {p['inflection_nm']} nm",
        f"lower bound: {p['lower_bound']}",
        f"upper bound: {p['upper_bound']}",
        "radius (nm) along the axis: " + ", ".join(f"{s} nm: {r:.3f}" for s, r in f["radius_nm"].items()),
        "",
    ])
