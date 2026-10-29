"""A PyMOL script that renders the G1/G2 atoms on the bound complex (``oneopes render-script``).

The script uses PyMOL's Python API, not a .pml command file: some PyMOL builds fail to parse
``set name, value`` lines in .pml files, and skip the setting with only a traceback.
"""
import logging
from pathlib import Path
from typing import Annotated

from cyclopts import Parameter
from cyclopts.types import ExistingFile

from oneopes.ligand.g1_g2 import AtomRef, read_g1_g2_yaml
from oneopes.render import G1_RGB, G2_RGB, Rgb

logger = logging.getLogger(__name__)

SCRIPT_TEMPLATE = """\
# Render of the G1/G2 atoms, written by `oneopes render-script`. Run with: pymol -cq {script}
from pymol import cmd

cmd.reinitialize()
cmd.load({complex_file!r}, "complex")
cmd.remove("solvent or inorganic")
cmd.hide("everything")
cmd.bg_color("white")
cmd.set("ray_opaque_background", 1)
cmd.set("antialias", 2)
cmd.set("ray_trace_mode", 1)
cmd.set("ray_trace_gain", 0.4)
cmd.set("ray_shadows", 0)
cmd.set("cartoon_fancy_helices", 1)

cmd.select("protein_atoms", "polymer.protein")
cmd.dss("protein_atoms")
cmd.show("cartoon", "protein_atoms")
cmd.color("grey80", "protein_atoms")
cmd.set("cartoon_transparency", 0.65, "protein_atoms")

cmd.select("ligand", "resn {ligand_resname}")
cmd.show("sticks", "ligand")
cmd.color("yellow", "ligand and elem C")
cmd.set("stick_radius", 0.16, "ligand")

cmd.select("pocket", "byres (polymer.protein within {pocket_cutoff_A} of ligand)")
cmd.show("sticks", "pocket")
cmd.color("grey60", "pocket and elem C")
cmd.set("stick_radius", 0.08, "pocket")
cmd.set("stick_transparency", 0.35, "pocket")

cmd.set_color("g1_color", {g1_color})
cmd.set_color("g2_color", {g2_color})
cmd.select("g1", "ligand and name {g1_names}")
cmd.select("g2", "ligand and name {g2_names}")
cmd.color("g1_color", "g1")
cmd.color("g2_color", "g2")
cmd.show("spheres", "g1 or g2")
cmd.set("sphere_scale", 0.55, "g1 or g2")
cmd.set("label_size", 22)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("label_font_id", 7)
cmd.set("label_position", (0, 2.2, 0))
cmd.label("g1 or g2", "name")

cmd.orient("ligand")
cmd.zoom("ligand or pocket", buffer=3.0)
cmd.turn("y", 20)
cmd.turn("x", -10)
cmd.png({image!r}, width=2400, height=1800, dpi=300, ray=1)
"""


def pymol_color(rgb: Rgb) -> str:
    """``rgb`` as PyMOL's [r, g, b] with components from 0 to 1."""
    return "[" + ", ".join(f"{component / 255:.3f}" for component in rgb) + "]"


def pymol_names(atoms: tuple[AtomRef, ...]) -> str:
    """The atom names as a PyMOL name list, e.g. C1+O4."""
    return "+".join(atom.name for atom in atoms)


def render_script(*, g1_g2: ExistingFile, complex_file: Annotated[ExistingFile, Parameter(name="--complex")],
                  ligand_resname: str, out: Path, image: Path = Path("selected_atoms_raw.png"),
                  pocket_cutoff_A: Annotated[float, Parameter(name="--pocket-cutoff-A")] = 5.0) -> None:
    """Write a PyMOL script that renders the G1/G2 atoms on the bound complex.

    The protein is a transparent cartoon, the ligand yellow sticks, and the G1/G2 atoms red and blue
    spheres, the colours `oneopes add-legend` uses. Run the script with `pymol -cq <out>`.

    Parameters
    ----------
    g1_g2
        g1_g2.yaml written by `oneopes select` or `oneopes monitor`.
    complex_file
        Bound protein-ligand structure, with the ligand atom names of g1_g2.yaml.
    ligand_resname
        Residue name of the ligand in the complex.
    out
        The PyMOL Python script to write, e.g. render.py.
    image
        The PNG the script renders, for `oneopes add-legend --src`. The script holds absolute paths, so it
        runs from any directory.
    pocket_cutoff_A
        Protein residues within this distance (A) of the ligand are drawn as thin sticks.
    """
    selection = read_g1_g2_yaml(g1_g2)
    script = SCRIPT_TEMPLATE.format(
        script=out.name, complex_file=str(complex_file.resolve()), ligand_resname=ligand_resname,
        pocket_cutoff_A=pocket_cutoff_A, g1_color=pymol_color(G1_RGB), g2_color=pymol_color(G2_RGB),
        g1_names=pymol_names(selection.G1), g2_names=pymol_names(selection.G2), image=str(image.resolve()),
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(script)
    logger.info("wrote %s; render with: pymol -cq %s", out, out)
