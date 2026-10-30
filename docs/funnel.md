# Funnel CV (`oneopes.funnel`)

A Python API around the public BioSimSpace funnel API (no `oneopes` command uses it yet).
BioSimSpace's `makeFunnel()` picks the P0/P1 atoms and `Funnel` defines the CV; this package
only derives P0/P1 (the mean position of each atom set) and exposes the CV as plain numbers.

```python
from oneopes.funnel.bss import read_system
from oneopes.funnel.geometry import funnel_extent, funnel_wall_points
from oneopes.funnel.make_funnel import make_funnel

system = read_system("top.top", "npt.gro", structure="whole.pdb")   # positions from whole.pdb (optional)
f = make_funnel(system)                     # extra kwargs go to BSS Funnel(...)
f.axis.atoms0, f.axis.p0, f.axis.direction  # makeFunnel atoms (0-based system indices), P0 (A), unit vector
f.parameters                                # width, buffer, steepness, inflection, bounds (nm)
funnel_extent(f.cv, [0.5, 1.0])             # radius (nm), from Funnel.getExtent()
funnel_wall_points(f.axis, f.cv)            # (N, 3) points on the funnel wall (A)
f.cv.getCorrection()                        # anything else: the BioSimSpace Funnel itself
```

`whole.pdb` from `gmx trjconv` has the positions but no molecule boundaries, which `makeFunnel()`
needs to find the protein and ligand; `read_system` takes the molecules from the topology and
the positions from the structure (rounded to 0.01 Å, through a `.gro`). `molecule_sizes(system)`
shows which molecules those are. See `notebooks/funnel_check.py`.

Lengths are in Å, CV parameters in nm. BioSimSpace is conda-only; without it the tests in
`tests/funnel/` are skipped. They find the validation systems in the repository's `original/`
tree or in `ONEOPES_FUNNEL_REF_DIRS`.
