"""
Publication-quality 3D render (UCSF Chimera) of a bound protein-ligand
complex, highlighting the G1 (anti-bulk) and G2 (bulk) atom pairs produced
by `ligand_waterfp.g1_g2_selection.select_g1_g2`.

Takes the structure, ligand residue name, and G1/G2 atom names entirely as
arguments - nothing about a specific system or ligand is hardcoded here.

Legacy Chimera's Python scripting does not support argparse-over-sys.argv
cleanly when launched via --script, so arguments are passed as simple
key=value tokens instead:

    chimera --nogui --silent --script "render_selected_atoms_chimera.py \\
        pdb=complex.pdb ligand_resname=MOL g1_atoms=ATOM1,ATOM2 g2_atoms=ATOM3,ATOM4 \\
        out_raw=selected_atoms_raw.png"
"""
import os
import sys
from chimera import runCommand as rc


def parse_kv_args():
    defaults = {
        "ligand_resname": "MOL",
        "pocket_cutoff": "5",
        "g1_color": "red",
        "g2_color": "blue",
        "out_raw": "selected_atoms_raw.png",
    }
    for tok in sys.argv[1:]:
        if "=" in tok:
            key, val = tok.split("=", 1)
            defaults[key] = val
    for required in ("pdb", "g1_atoms", "g2_atoms"):
        if required not in defaults:
            raise SystemExit(f"Missing required argument '{required}=...' "
                              "(see this script's docstring for usage)")
    return defaults


args = parse_kv_args()
g1_spec = "@" + ",".join(a.strip() for a in args["g1_atoms"].split(","))
g2_spec = "@" + ",".join(a.strip() for a in args["g2_atoms"].split(","))
ligand_resname = args["ligand_resname"]
pocket_cutoff = args["pocket_cutoff"]

rc("open " + args["pdb"])
rc("delete :SOL,HOH,WAT,TIP3,NA,CL,K")

rc("background solid white")
rc("~display")
rc("~ribbon")

# protein: transparent ribbon for context
rc("ribbon protein")
rc("color light gray protein")
rc("transparency 80,r protein")

# pocket residues within the cutoff of the ligand, thin sticks (heavy atoms
# only), no labels - context only, kept subdued so the ligand stands out
rc(f"select :{ligand_resname} z<{pocket_cutoff} & protein & ~@/element=H")
rc("display sel")
rc("repr stick sel")
rc("color gray70 sel")
rc("setattr m stickScale 0.3 sel")

# ligand itself: distinct carbon color (heavy atoms only) so it reads as a
# clearly separate, non-protein entity
rc(f"display :{ligand_resname} & ~@/element=H")
rc(f"repr stick :{ligand_resname}")
rc(f"color byelement :{ligand_resname}")
rc(f"color yellow :{ligand_resname} & @/element=C")
rc(f"setattr m stickScale 0.75 :{ligand_resname}")

# G1 (anti-bulk) atoms
rc(f"select :{ligand_resname}{g1_spec}")
rc(f"color {args['g1_color']} sel")
rc("repr sphere sel")
rc("setattr a radius 0.85 sel")
rc("label sel")
rc("setattr a labelColor black sel")

# G2 (bulk) atoms
rc(f"select :{ligand_resname}{g2_spec}")
rc(f"color {args['g2_color']} sel")
rc("repr sphere sel")
rc("setattr a radius 0.85 sel")
rc("label sel")
rc("setattr a labelColor black sel")

# camera
rc(f"focus :{ligand_resname} z<{pocket_cutoff}")
rc("turn y 25")
rc("turn x -12")

rc("windowsize 2400 1800")
rc("copy file " + args["out_raw"] + " width 2400 height 1800 supersample 3")

print("[render_selected_atoms_chimera] wrote raw render to " + os.path.abspath(args["out_raw"]))
rc("stop now")
