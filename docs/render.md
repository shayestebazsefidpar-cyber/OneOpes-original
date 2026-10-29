# Stage 5 - render (optional)

Draws the G1/G2 atoms on the bound complex: protein as a transparent cartoon, ligand as sticks,
G1 red and G2 blue spheres. The render scripts in `src/oneopes/render/` run inside PyMOL or
legacy UCSF Chimera, so they take the atom names as arguments.

```bash
pymol -cq render_selected_atoms.py -- --pdb complex.pdb --ligand-resname MOL \
    --g1-atoms A1,A2 --g2-atoms A3,A4 --out-raw raw.png
# or: chimera --nogui --silent --script "render_selected_atoms_chimera.py pdb=complex.pdb \
#     ligand_resname=MOL g1_atoms=A1,A2 g2_atoms=A3,A4 out_raw=raw.png"

oneopes add-legend --src raw.png --dst selected_atoms.png --g1-g2 g1_g2.yaml \
    --caption "grey = protein pocket, yellow = ligand"
```

`--g1-g2` labels the G1/G2 entries; `--entry "title:detail:r,g,b"` adds more.
