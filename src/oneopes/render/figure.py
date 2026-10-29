"""The figure of the selected atoms on the bound complex (``oneopes render``)."""
import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingFile
from matplotlib.image import imread

from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import G1G2Selection, read_g1_g2_yaml
from oneopes.render import RENDER_DPI
from oneopes.render.legend import with_legend
from oneopes.render.pymol_script import pymol_script, run_pymol
from oneopes.system import load_universe

logger = logging.getLogger(__name__)


def require_atoms(complex_file: Path, ligand_resname: str, selection: G1G2Selection) -> None:
    """Raise InputError unless the complex has the ligand and its G1/G2 atoms, which PyMOL would skip silently."""
    ligand_names = set(load_universe(complex_file).select_atoms(f"resname {ligand_resname}").names)
    if not ligand_names:
        raise InputError(f"{complex_file.name} has no residue named {ligand_resname}.")
    missing = [atom.name for atom in (*selection.G1, *selection.G2) if atom.name not in ligand_names]
    if missing:
        raise InputError(f"The {ligand_resname} residue of {complex_file.name} has no atoms named {missing}.")


def render(*, g1_g2: ExistingFile, complex_file: Annotated[ExistingFile, Parameter(name="--complex")],
           ligand_resname: str, out: Path, caption: str = "", font_size: float = 8,
           pocket_cutoff_A: Annotated[float, Parameter(name="--pocket-cutoff-A")] = 5.0,
           pymol: str = "pymol") -> None:
    """Render the G1/G2 atoms on the bound complex with PyMOL, and add their legend.

    The protein is a transparent cartoon, the ligand yellow sticks, and the G1/G2 atoms red and blue
    spheres. Next to --out go the render without the legend (<name>_raw.png) and the PyMOL script
    (<name>_pymol.py), to edit and run again with `pymol -cq`.

    Parameters
    ----------
    g1_g2
        g1_g2.yaml written by `oneopes select` or `oneopes monitor`.
    complex_file
        Bound protein-ligand structure, with the ligand atom names of g1_g2.yaml.
    ligand_resname
        Residue name of the ligand in the complex.
    out
        The figure to write (PNG).
    caption
        Text along the bottom of the figure (default: none).
    font_size
        Font size (pt) of the legend and the caption.
    pocket_cutoff_A
        Protein residues within this distance (A) of the ligand are drawn as thin sticks.
    pymol
        PyMOL executable.
    """
    selection = read_g1_g2_yaml(g1_g2)
    require_atoms(complex_file, ligand_resname, selection)
    raw = out.with_name(f"{out.stem}_raw.png")
    script = out.with_name(f"{out.stem}_pymol.py")
    out.parent.mkdir(parents=True, exist_ok=True)
    script.write_text(pymol_script(selection, complex_file, ligand_resname, raw, pocket_cutoff_A))
    raw.unlink(missing_ok=True)
    run_pymol(pymol, script, raw)
    with_legend(imread(raw), selection, caption, font_size).savefig(out, dpi=RENDER_DPI)
    logger.info("wrote %s, %s and %s", out, raw.name, script.name)
