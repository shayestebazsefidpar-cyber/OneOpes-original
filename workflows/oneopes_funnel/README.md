# oneopes_funnel

A thin, generic OneOPES API around the **public BioSimSpace** funnel API
(`BioSimSpace.Metadynamics.CollectiveVariable.makeFunnel` and `Funnel`).
BioSimSpace selects the P0/P1 atoms and defines the funnel CV; this package
does not implement its own P0/P1 selection.

Status: `p0_p1.py`, `geometry.py`, `make_funnel.py` implemented;
`system_io.py` and `plumed_io.py` are still stubs.

## What comes from where

| Quantity | Source |
|---|---|
| `atoms0` / `atoms1` | BioSimSpace `makeFunnel()`, called unchanged |
| P0 / P1, axis, axis length | **derived** here from the atoms `makeFunnel()` selected (average position, A); `makeFunnel()` itself returns atom selections only |
| CV parameters | BioSimSpace `Funnel` getters |
| funnel radius | BioSimSpace `Funnel.getExtent()` |
| entropy correction | BioSimSpace `Funnel.getCorrection()`, via `.cv` |
| wall points | `funnel_wall_points()` (this package): rings around the P0 -> P1 axis, every `step_A` between the CV's lower and upper bounds, radius from `getExtent()` |

Only public BioSimSpace modules and functions are used (no `_`-prefixed
modules or attributes).

## Conventions

- Coordinates and lengths in Angstrom; CV parameters in nm (BioSimSpace's unit
  for the CV).
- `atoms0` / `atoms1` are 0-based **system-level** atom indices, as returned
  by `makeFunnel()`.

## Usage

```python
import BioSimSpace as BSS
from oneopes_funnel.make_funnel import make_funnel

system = BSS.IO.readMolecules(["top.top", "npt.gro"])
f = make_funnel(system)               # extra kwargs go to BSS Funnel(...)
f.atoms0, f.atoms1                    # BSS makeFunnel selections
f.p0, f.p1, f.axis, f.axis_length     # derived from those selections (A)
f.parameters                          # width, buffer, steepness, inflection, bounds (nm)
f.extent_nm([0.5, 1.0, 2.0])          # radius via Funnel.getExtent()
f.wall_points(step_A=2.0, n_angles=8) # (N, 3) wall points in A
f.cv                                  # the BioSimSpace Funnel object (e.g. f.cv.getCorrection())
```

## Install and test

BioSimSpace is conda-only (not pip-installable); use a conda env that has it.

```bash
pip install -e ".[dev]"
pytest
```

The tests compare against the installed BioSimSpace through its public API.
Without BioSimSpace, BioSimSpace-dependent tests are skipped. Validation
systems (HSP90/lig1, BRD4/lig1) are found in the repository's `original/`
tree, or set `ONEOPES_FUNNEL_REF_DIRS`.

## Layout

| Module | Purpose |
|---|---|
| `p0_p1.py` | `makeFunnel()` selections + derived P0/P1 |
| `geometry.py` | CV parameters, `getExtent()` radius, wall points |
| `make_funnel.py` | `make_funnel()` - selections + the BioSimSpace `Funnel` CV |
| `plumed_io.py` | (stub) read-only parsing of existing `plumed.dat` files |
| `system_io.py` | (stub) system loading |

Repository location: `workflows/oneopes_funnel/` in `OneOpes-original`.
