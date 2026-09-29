# oneopes_funnel

A thin, generic OneOPES API around the **BioSimSpace** funnel implementation
(BioSimSpace 2024.4.1, `BioSimSpace/Metadynamics/CollectiveVariable/_funnel.py`).
BioSimSpace is the source of truth; this package does not implement its own
P0/P1 or funnel algorithm.

Status: `p0_p1.py`, `geometry.py`, `make_funnel.py` implemented;
`system_io.py` and `plumed_io.py` are still stubs.

## What comes from BioSimSpace

| Quantity | Source |
|---|---|
| `atoms0` / `atoms1` | BSS `makeFunnel()`, called unchanged |
| P0 / P1, axis, axis length | **derived** from the atoms `makeFunnel()` selected (average position, A) - `makeFunnel()` itself returns atom selections only |
| CV parameters | BSS `Funnel` object getters |
| funnel radius | BSS `Funnel.getExtent()` (the same formula BSS writes to PLUMED) |
| entropy correction | BSS `Funnel.getCorrection()`, via `.cv` |
| wall points | the BSS `viewFunnel()` ring loop (lines 1098-1149), copied verbatim with attribution because `viewFunnel` returns no coordinates |

`makeFunnel()` in short: protein = largest molecule, ligand = second-largest;
`atoms1` = CA atoms within 10 A of the ligand COM; the open side is found from
a 5x5x5 grid (20 A) of points with no protein atom within 2 A; `atoms0` = CA
atoms within 7 A of a point 10 A from the atoms1 centroid, away from the open
side.

## Conventions

- Coordinates and lengths in Angstrom; CV parameters in nm (BSS's internal unit).
- `atoms0` / `atoms1` are 0-based **system-level** atom indices, as returned by
  `makeFunnel()`. BSS's PLUMED writer uses `index + 1` and labels them
  `p1` (= atoms0, funnel origin) and `p2` (= atoms1).

## Deliberate differences from BSS `viewFunnel()` (visualization only)

`viewFunnel()` is a notebook visualization helper and disagrees with the
Funnel CV in two places; this package follows the CV/PLUMED behavior by
default:

1. P0/P1 use system-level indices. `viewFunnel` looks the same indices up in
   the protein molecule's own atom list, which differs when the protein is not
   the first molecule in the system.
2. `wall_points(radius="cv")` (default) uses `getExtent()`.
   `radius="viewfunnel"` reproduces viewFunnel's own radius line, which puts
   Angstrom distances into a per-nm steepness.

## Usage

```python
import BioSimSpace as BSS
from oneopes_funnel.make_funnel import make_funnel

system = BSS.IO.readMolecules(["top.top", "npt.gro"])
f = make_funnel(system)            # extra kwargs go to BSS Funnel(...)
f.atoms0, f.atoms1                 # BSS makeFunnel selections
f.p0, f.p1, f.axis, f.axis_length  # derived from those selections (A)
f.parameters                       # width, buffer, steepness, inflection, bounds (nm)
f.extent_nm([0.5, 1.0, 2.0])       # radius via Funnel.getExtent()
f.wall_points()                    # (N, 3) wall points in A
f.cv                               # the BioSimSpace Funnel object (e.g. f.cv.getCorrection())
```

## Install and test

BioSimSpace is conda-only (not pip-installable); use a conda env that has
BioSimSpace 2024.4.1.

```bash
pip install -e ".[dev]"
pytest
```

The test suite compares against the installed BioSimSpace directly and
**fails** if it is not version 2024.4.1 or if the reviewed BSS source files
changed. Without BioSimSpace, BSS-dependent tests are skipped. Validation
systems (HSP90/lig1, BRD4/lig1) are found in the repository's `original/`
tree, or set `ONEOPES_FUNNEL_REF_DIRS`.

## License / attribution

The funnel science is BioSimSpace's (GPL-3.0-or-later, (c) 2017-2025 Lester
Hedges; `makeFunnel`/`viewFunnel` adapted there from `funnel_maker.py` by
Dominykas Lukauskis). This package imports BioSimSpace at runtime and contains
a verbatim copy of part of `viewFunnel()` (see the attribution in
`src/oneopes_funnel/geometry.py`), so it is licensed under GPL-3.0-or-later;
see `LICENSE`.

## Layout

| Module | Purpose |
|---|---|
| `p0_p1.py` | `makeFunnel()` selections + derived P0/P1 |
| `geometry.py` | CV parameters, `getExtent()` radius, viewFunnel wall points |
| `make_funnel.py` | `make_funnel()` - selections + the BSS `Funnel` CV |
| `plumed_io.py` | (stub) read-only parsing of existing `plumed.dat` files |
| `system_io.py` | (stub) system loading |

Repository location: `workflows/oneopes_funnel/` in `OneOpes-original`.
