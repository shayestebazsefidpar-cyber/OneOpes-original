# Stage 5 - render (optional)

`oneopes render` draws the G1/G2 atoms of `g1_g2.yaml` on the bound complex with PyMOL: protein
as a transparent cartoon, ligand as yellow sticks, G1 red and G2 blue spheres, with a legend in
the same colours.

```bash
oneopes render --g1-g2 g1_g2.yaml --complex complex.pdb --ligand-resname MOL --out selected_atoms.png \
    --caption "grey = protein pocket, yellow = ligand"
```

Next to the figure it writes the render without the legend (`selected_atoms_raw.png`) and the
PyMOL script (`selected_atoms_pymol.py`), to adjust the view and run again with `pymol -cq`.
PyMOL must be on `PATH`, or given with `--pymol`. The complex must use the ligand atom names of
`g1_g2.yaml`.
