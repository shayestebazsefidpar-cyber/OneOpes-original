# Stage 5 - Visualization

Renders the G1/G2 selected atoms on the bound complex for a quick visual
sanity check / figure. Optional - Stages 1-4 form the complete
methodology; this stage is for communicating the result.

- **`render_selected_atoms.py`** (PyMOL): loads a bound-complex structure,
  shows the protein as a transparent cartoon, the ligand as sticks, and
  the G1/G2 atoms as colored labeled spheres.
- **`render_selected_atoms_chimera.py`**: the same render, for UCSF
  Chimera (legacy) instead of PyMOL.
- **`add_legend.py`**: composites a legend box (and optional caption) onto
  either raw render.

The renders take the structure path, ligand residue name, and G1/G2 atom
names purely as arguments - nothing about a specific system or ligand is
hardcoded; `add-legend` reads the G1/G2 names from `g1_g2.yaml`.

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
oneopes add-legend \
    --src outputs/visualization/selected_atoms_raw.png \
    --dst outputs/visualization/selected_atoms.png \
    --title "WaterFP selected atoms" \
    --g1-g2 outputs/convergence/selection/g1_g2.yaml \
    --caption "grey = protein pocket, yellow = ligand"
```

`--g1-g2` gives the G1 (red) and G2 (blue) entries their atom names; add
more swatches, or replace them, with `--entry "title:detail:r,g,b"`.

Rendered images are scientific output for a specific system - write them
under `outputs/` (git-ignored), never commit a PNG into this folder.
