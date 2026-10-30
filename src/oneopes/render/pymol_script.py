"""The PyMOL script that renders the G1/G2 atoms on the bound complex, and running it.

The script uses PyMOL's Python API, not a .pml command file: some PyMOL builds fail to parse
``set name, value`` lines in .pml files, and skip the setting with only a traceback.
"""
import subprocess
from pathlib import Path

from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import AtomRef, G1G2Selection
from oneopes.render import G1_RGB, G2_RGB, RENDER_DPI, RENDER_SIZE, Rgb

SCRIPT_TEMPLATE = """\
# Render of the G1/G2 atoms, written by `oneopes render`. Run again with: pymol -cq <this file>
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

cmd.select("ligand", "id {ligand_ids}")
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
cmd.png({image!r}, width={width}, height={height}, dpi={dpi}, ray=1)
"""


def pymol_color(rgb: Rgb) -> str:
    """``rgb`` as PyMOL's [r, g, b] with components from 0 to 1."""
    return "[" + ", ".join(f"{component / 255:.3f}" for component in rgb) + "]"


def pymol_names(atoms: tuple[AtomRef, ...]) -> str:
    """The atom names as a PyMOL name list, e.g. C1+O4."""
    return "+".join(atom.name for atom in atoms)


def pymol_script(selection: G1G2Selection, complex_file: Path, ligand_ids: list[int], image: Path,
                 pocket_cutoff_A: float) -> str:
    """The script rendering ``selection`` on the complex into ``image``; with absolute paths, it runs anywhere.

    ``ligand_ids`` are the ligand's atom ids (serial numbers) in ``complex_file``, which PyMOL keeps as ``id``.
    """
    width, height = RENDER_SIZE
    return SCRIPT_TEMPLATE.format(
        complex_file=str(complex_file.resolve()), ligand_ids="+".join(map(str, ligand_ids)),
        pocket_cutoff_A=pocket_cutoff_A,
        g1_color=pymol_color(G1_RGB), g2_color=pymol_color(G2_RGB),
        g1_names=pymol_names(selection.G1), g2_names=pymol_names(selection.G2),
        image=str(image.resolve()), width=width, height=height, dpi=RENDER_DPI,
    )


def run_pymol(pymol: str, script: Path, image: Path) -> None:
    """Run ``script`` in PyMOL; raises InputError, with the end of PyMOL's output, unless it wrote ``image``."""
    try:
        result = subprocess.run([pymol, "-cq", str(script)], capture_output=True, text=True)
    except FileNotFoundError:
        raise InputError(f"PyMOL not found as {pymol!r}; pass its path with --pymol.") from None
    if not image.exists():
        output = "\n".join((result.stdout + result.stderr).splitlines()[-20:])
        raise InputError(f"PyMOL did not write {image.name}; the end of its output:\n{output}")
