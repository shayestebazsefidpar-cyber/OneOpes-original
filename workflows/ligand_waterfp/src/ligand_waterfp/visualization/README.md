# Stage 7 - Visualization

Renders the G1/G2 selected atoms on the bound complex for a quick visual
sanity check / figure. Optional - Stages 1-6 form the complete
methodology; this stage is for communicating the result.

- **`render_selected_atoms.py`** (PyMOL): loads a bound-complex structure,
  shows the protein as a transparent cartoon, the ligand as sticks, and
  the G1/G2 atoms as colored labeled spheres.
- **`render_selected_atoms_chimera.py`**: the same render, for UCSF
  Chimera (legacy) instead of PyMOL.
- **`add_legend.py`**: composites a legend box (and optional caption) onto
  either raw render.

All three take the structure path, ligand residue name, and G1/G2 atom
names purely as arguments - nothing about a specific system or ligand is
hardcoded.

## Usage

```bash
# PyMOL
pymol -cq render_selected_atoms.py -- \
    --pdb complex.pdb --ligand-resname MOL \
    --g1-atoms <name1>,<name2> --g2-atoms <name3>,<name4> \
    --out-raw outputs/visualization/selected_atoms_raw.png

# or Chimera
chimera --nogui --silent --script "render_selected_atoms_chimera.py \
    pdb=complex.pdb ligand_resname=MOL g1_atoms=<name1>,<name2> g2_atoms=<name3>,<name4> \
    out_raw=outputs/visualization/selected_atoms_raw.png"

# then, either way:
ligand-waterfp-add-legend \
    --src outputs/visualization/selected_atoms_raw.png \
    --dst outputs/visualization/selected_atoms.png \
    --title "WaterFP selected atoms" \
    --entry "Anti-bulk (G1):<name1>, <name2>:220,20,20" \
    --entry "Bulk (G2):<name3>, <name4>:30,60,220" \
    --caption "grey = protein pocket, yellow = ligand"
```

Rendered images are scientific output for a specific system - write them
under `outputs/` (git-ignored), never commit a PNG into this folder.
