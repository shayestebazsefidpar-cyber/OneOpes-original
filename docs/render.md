# Stage 5 - render (optional)

Draws the G1/G2 atoms on the bound complex: protein as a transparent cartoon, ligand as yellow
sticks, G1 red and G2 blue spheres. `oneopes render-script` writes a PyMOL script with the atoms
of `g1_g2.yaml`; PyMOL renders it; `oneopes add-legend` adds a legend in the same colours.

```bash
oneopes render-script --g1-g2 g1_g2.yaml --complex complex.pdb --ligand-resname MOL --out render.py
pymol -cq render.py                       # writes selected_atoms_raw.png (--image)
oneopes add-legend --src selected_atoms_raw.png --dst selected_atoms.png --g1-g2 g1_g2.yaml \
    --caption "grey = protein pocket, yellow = ligand"
```

The complex must use the ligand atom names of `g1_g2.yaml`. `--entry "title:detail:r,g,b"` adds
more legend entries.
