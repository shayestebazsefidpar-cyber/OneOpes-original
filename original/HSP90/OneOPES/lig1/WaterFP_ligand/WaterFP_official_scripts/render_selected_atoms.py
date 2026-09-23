"""
Publication-quality 3D render of the HSP90-lig1 bound complex, highlighting
the four ligand atoms selected by the official WaterFP selection procedure
(see RESULTS_SUMMARY.md in this same directory):
  anti-bulk pair: N1, C8
  bulk pair:      O3, O2

Structure: original/HSP90/OneOPES/lig1/whole.pdb (the actual bound complex,
not the ligand-only WaterFP_ligand system). Read-only input; nothing in
original/HSP90/OneOPES/ is modified.

Run with: pymol -cq render_selected_atoms.py
"""
import os
from pymol import cmd

PDB = "/home/grheco/Escritorio/work/software/oneopes/OneOpes-original/original/HSP90/OneOPES/lig1/whole.pdb"
OUT_RAW = "HSP90_lig1_WaterFP_selected_atoms_raw.png"
OUT_FINAL = "HSP90_lig1_WaterFP_selected_atoms.png"

cmd.reinitialize()
cmd.load(PDB, "complex")

# drop solvent/ions - protein/pocket + ligand figure only
cmd.remove("resn SOL+NA+CL")

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
cmd.select("lig", "resn MOL")
cmd.show("sticks", "lig")
cmd.color("yellow", "lig and elem C")
cmd.set("stick_radius", 0.16, "lig")

# pocket residues for spatial context (within 5 A of ligand), thin lines, no labels
cmd.select("pocket", "byres (polymer.protein within 5 of lig)")
cmd.show("sticks", "pocket")
cmd.color("grey60", "pocket and elem C")
cmd.set("stick_radius", 0.08, "pocket")
cmd.set("stick_transparency", 0.35, "pocket")

# the four selected atoms
cmd.select("antibulk", "lig and name N1+C8")
cmd.select("bulk", "lig and name O3+O2")

ANTIBULK_COLOR = "red"
BULK_COLOR = "blue"

cmd.color(ANTIBULK_COLOR, "antibulk")
cmd.color(BULK_COLOR, "bulk")
cmd.show("spheres", "antibulk or bulk")
cmd.set("sphere_scale", 0.55, "antibulk or bulk")
cmd.set("sphere_transparency", 0.0, "antibulk or bulk")

# labels
cmd.set("label_size", 22)
cmd.set("label_color", "black")
cmd.set("label_outline_color", "white")
cmd.set("label_font_id", 7)  # sans bold
cmd.label("antibulk", '"%s" % name')
cmd.label("bulk", '"%s" % name')

cmd.set("label_position", (0, 2.2, 0))

# view: center on ligand + pocket, zoomed in enough to see the four atoms clearly
cmd.orient("lig")
cmd.zoom("lig or pocket", buffer=3.0)
cmd.turn("y", 20)
cmd.turn("x", -10)

cmd.set("ray_shadows", 0)
cmd.ray(2400, 1800)
cmd.png(OUT_RAW, width=2400, height=1800, dpi=300, ray=1)

print(f"[render_selected_atoms] wrote raw render to {os.path.abspath(OUT_RAW)}")
