"""
Publication-quality 3D render of the HSP90-lig1 bound complex (UCSF Chimera),
highlighting the four ligand atoms selected by the official WaterFP selection
procedure (see RESULTS_SUMMARY.md in this directory):
  anti-bulk pair: N1, C8
  bulk pair:      O3, O2

Structure: original/HSP90/OneOPES/lig1/whole.pdb (the actual bound complex,
not the ligand-only WaterFP_ligand system). Read-only input.

Run with: chimera --nogui --silent --script render_selected_atoms_chimera.py
"""
import os
from chimera import runCommand as rc

PDB = "/home/grheco/Escritorio/work/software/oneopes/OneOpes-original/original/HSP90/OneOPES/lig1/whole.pdb"
OUTDIR = "/home/grheco/Escritorio/work/software/oneopes/OneOpes-original/original/HSP90/OneOPES/lig1/WaterFP_ligand/WaterFP_official_scripts"
OUT_RAW = os.path.join(OUTDIR, "HSP90_lig1_WaterFP_selected_atoms_raw.png")

rc("open " + PDB)
rc("delete :SOL,NA,CL")

rc("background solid white")
rc("~display")
rc("~ribbon")

# protein: transparent ribbon for context
rc("ribbon protein")
rc("color light gray protein")
rc("transparency 80,r protein")

# pocket residues within 5 A of the ligand, thin sticks (heavy atoms only),
# no labels - context only, kept subdued so the ligand stands out
rc("select :MOL z<5 & protein & ~@/element=H")
rc("display sel")
rc("repr stick sel")
rc("color gray70 sel")
rc("setattr m stickScale 0.3 sel")

# ligand itself: distinct carbon color (heavy atoms only) so it reads as a
# clearly separate, non-protein entity
rc("display :MOL & ~@/element=H")
rc("repr stick :MOL")
rc("color byelement :MOL")
rc("color yellow :MOL & @/element=C")
rc("setattr m stickScale 0.75 :MOL")

# the four selected atoms - act on each selection immediately via the
# reusable "sel" keyword, since Chimera's `select` does not support naming
# a selection inline (that syntax is a different, unrelated "mangled spec").
rc("select :MOL@N1,C8")
rc("color red sel")
rc("repr sphere sel")
rc("setattr a radius 0.85 sel")
rc("label sel")
rc("setattr a labelColor black sel")

rc("select :MOL@O3,O2")
rc("color blue sel")
rc("repr sphere sel")
rc("setattr a radius 0.85 sel")
rc("label sel")
rc("setattr a labelColor black sel")

# camera
rc("focus :MOL z<5")
rc("turn y 25")
rc("turn x -12")

rc("windowsize 2400 1800")
rc("copy file " + OUT_RAW + " width 2400 height 1800 supersample 3")

print("[render_selected_atoms_chimera] wrote raw render to " + OUT_RAW)
rc("stop now")
