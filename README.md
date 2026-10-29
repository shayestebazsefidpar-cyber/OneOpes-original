# oneopes

Builds ligand-side hydration collective variables for OneOPES from unbiased MD:
ligand-in-water MD → block-wise WaterFP convergence → G1/G2 atom pairs → PLUMED fragment.
It also analyses protein atoms around a bound ligand ([docs/protein.md](docs/protein.md))
and wraps the BioSimSpace funnel CV ([docs/funnel.md](docs/funnel.md)).

Code only: no systems or results are kept here. Write outputs under `outputs/` (git-ignored).

## Install

```bash
pip install -e ".[dev]"   # then: pytest
```

Also needed: GROMACS on `PATH` (stage 1); PyMOL only for the renders;
BioSimSpace (conda) only for `oneopes.funnel`.

## Run

```bash
oneopes prepare-system --ligand-gro ligand.gro --ligand-top ligand.top --outdir outputs/lig_system
# start production MD (docs/ligand_setup.md), then follow it until hydration converges:
oneopes monitor $MDRUN_PID --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence
oneopes build-cv --g1-g2 outputs/convergence/selection/g1_g2.yaml --out outputs/plumed_fragment.dat
```

The monitor stops mdrun on convergence and writes the G1/G2 selection of the converged
block. Put per-system settings in a YAML file (see `configs/example.yaml`) and run
`oneopes --config my_system.yaml <command>`; command-line flags override it.
Every analysis command writes `run_metadata.json` (input hashes, options, versions).

| Command | Does | Docs |
|---|---|---|
| `prepare-system` | build and equilibrate the ligand-in-water system | [ligand_setup.md](docs/ligand_setup.md) |
| `monitor` | WaterFP per block until converged; stop mdrun; select G1/G2 | [ligand_monitor.md](docs/ligand_monitor.md) |
| `select`, `select-block` | G1/G2 over a frame window, or from a block of `blocks.nc` | [g1_g2.md](docs/g1_g2.md) |
| `build-cv` | PLUMED fragment for G1/G2 (review before use) | [plumed_cv.md](docs/plumed_cv.md) |
| `render-script`, `add-legend` | PyMOL script that renders G1/G2 on the complex; legend on the render | [render.md](docs/render.md) |
| `protein` | WaterFP, contacts and SASA of protein atoms near the ligand | [protein.md](docs/protein.md) |

Why the method works as it does: [docs/METHOD_RATIONALE.md](docs/METHOD_RATIONALE.md).

## Notebooks

[marimo](https://marimo.io) notebooks for interactive work (`pip install -e ".[notebooks]"`,
then `marimo edit notebooks/<name>.py`):

- `explore_run.py`: open a `blocks.nc` and look at its summary, FP over time, block comparisons
  and g(r) per block.
- `funnel_check.py`: rebuild a system's funnel axis with BioSimSpace (needs BioSimSpace).

## Code

Each command is a plain function; `cli.py` turns it into a subcommand with
[cyclopts](https://cyclopts.readthedocs.io): flags come from its parameters, help from its docstring.
