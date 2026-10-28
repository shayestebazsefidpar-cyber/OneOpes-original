"""
Publication-quality 3D render (PyMOL) of a bound protein-ligand complex,
highlighting the G1 (anti-bulk) and G2 (bulk) atom pairs produced by
`ligand_waterfp.g1_g2_selection.select_g1_g2`.

Takes the structure, ligand residue name, and G1/G2 atom names entirely as
arguments - nothing about a specific system or ligand is hardcoded here.

Run with:
    pymol -cq render_selected_atoms.py -- \\
        --pdb complex.pdb --ligand-resname MOL \\
        --g1-atoms ATOM1,ATOM2 --g2-atoms ATOM3,ATOM4 \\
        --out-raw selected_atoms_raw.png
"""
import argparse
import os
import sys
from pymol import cmd


def parse_args():
    # PyMOL passes everything after "--" through to sys.argv unchanged.
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else sys.argv[1:]
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--pdb", required=True, help="Bound complex structure (protein + ligand)")
    p.add_argument("--ligand-resname", default="MOL")
    p.add_argument("--g1-atoms", required=True, help="Comma-separated G1 (anti-bulk) atom names, e.g. ATOM1,ATOM2")
    p.add_argument("--g2-atoms", required=True, help="Comma-separated G2 (bulk) atom names, e.g. ATOM3,ATOM4")
    p.add_argument("--pocket-cutoff-A", type=float, default=5.0,
                    help="Show protein residues within this distance of the ligand for context (default: 5.0)")
    p.add_argument("--g1-color", default="red")
    p.add_argument("--g2-color", default="blue")
    p.add_argument("--out-raw", default="selected_atoms_raw.png")
    return p.parse_args(argv)


def main():
    args = parse_args()
    g1_names = "+".join(a.strip() for a in args.g1_atoms.split(","))
    g2_names = "+".join(a.strip() for a in args.g2_atoms.split(","))

    cmd.reinitialize()
    cmd.load(args.pdb, "complex")

    # drop common solvent/ion residue names - protein/pocket + ligand figure only
    cmd.remove("resn SOL+HOH+WAT+TIP3+TIP3P+NA+CL+K")

    cmd.hide("everything")
    cmd.bg_color("white")
    cmd.set("ray_opaque_background", 1)
    cmd.set("antialias", 2)
    cmd.set("ray_trace_mode", 1)
    cmd.set("ray_trace_gain", 0.4)
    cmd.set("cartoon_fancy_helices", 1)

    # protein: transparent cartoon for context
    cmd.select("prot", "polymer.protein")
    cmd.dss("prot")
    cmd.show("cartoon", "prot")
    cmd.color("grey80", "prot")
    cmd.set("cartoon_transparency", 0.65, "prot")

    # ligand
    cmd.select("lig", f"resn {args.ligand_resname}")
    cmd.show("sticks", "lig")
    cmd.color("yellow", "lig and elem C")
    cmd.set("stick_radius", 0.16, "lig")

    # pocket residues for spatial context, thin lines, no labels
    cmd.select("pocket", f"byres (polymer.protein within {args.pocket_cutoff_A} of lig)")
    cmd.show("sticks", "pocket")
    cmd.color("grey60", "pocket and elem C")
    cmd.set("stick_radius", 0.08, "pocket")
    cmd.set("stick_transparency", 0.35, "pocket")

    # the G1/G2 selected atoms
    cmd.select("g1", f"lig and name {g1_names}")
    cmd.select("g2", f"lig and name {g2_names}")

    cmd.color(args.g1_color, "g1")
    cmd.color(args.g2_color, "g2")
    cmd.show("spheres", "g1 or g2")
    cmd.set("sphere_scale", 0.55, "g1 or g2")
    cmd.set("sphere_transparency", 0.0, "g1 or g2")

    # labels
    cmd.set("label_size", 22)
    cmd.set("label_color", "black")
    cmd.set("label_outline_color", "white")
    cmd.set("label_font_id", 7)  # sans bold
    cmd.label("g1", '"%s" % name')
    cmd.label("g2", '"%s" % name')
    cmd.set("label_position", (0, 2.2, 0))

    # view: center on ligand + pocket, zoomed in enough to see the selected atoms clearly
    cmd.orient("lig")
    cmd.zoom("lig or pocket", buffer=3.0)
    cmd.turn("y", 20)
    cmd.turn("x", -10)

    cmd.set("ray_shadows", 0)
    cmd.ray(2400, 1800)
    cmd.png(args.out_raw, width=2400, height=1800, dpi=300, ray=1)

    print(f"[render_selected_atoms] wrote raw render to {os.path.abspath(args.out_raw)}")


main()
