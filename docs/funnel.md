# Funnel CV (`oneopes.funnel`)

A Python API around the public BioSimSpace funnel API (no `oneopes` command uses it yet).
BioSimSpace's `makeFunnel()` picks the P0/P1 atoms and `Funnel` defines the CV; this package
only derives P0/P1 (the mean position of each atom set) and exposes the CV as plain numbers.

```python
import BioSimSpace as BSS
from oneopes.funnel.geometry import funnel_extent, funnel_wall_points
from oneopes.funnel.make_funnel import make_funnel

f = make_funnel(BSS.IO.readMolecules(["top.top", "npt.gro"]))   # extra kwargs go to BSS Funnel(...)
f.p0p1.atoms0, f.p0p1.p0, f.p0p1.axis       # makeFunnel atoms (0-based system indices), P0, axis (A)
f.parameters                                # width, buffer, steepness, inflection, bounds (nm)
funnel_extent(f.cv, [0.5, 1.0])             # radius (nm), from Funnel.getExtent()
funnel_wall_points(f.p0p1, f.cv)            # (N, 3) points on the funnel wall (A)
f.cv.getCorrection()                        # anything else: the BioSimSpace Funnel itself
```

Lengths are in Å, CV parameters in nm. BioSimSpace is conda-only; without it the tests in
`tests/funnel/` are skipped. They find the validation systems in the repository's `original/`
tree or in `ONEOPES_FUNNEL_REF_DIRS`.
