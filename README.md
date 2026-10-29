# oneopes

A reusable, system-independent, **installable Python package** for
ligand-side hydration-CV construction:

```text
ligand-only unbiased MD
        |
WaterFP calculation
        |
block-wise convergence
        |
G1/G2 ligand atom selection
        |
ligand-side CV construction
```

It also holds the protein-side analysis around a bound ligand
(`oneopes protein`, see [docs/protein.md](docs/protein.md)) and a Python API
for the BioSimSpace funnel CV ([docs/funnel.md](docs/funnel.md)).

This is **code only** - no example systems, no HSP90/BRD4 results, no
trajectories, no committed scientific output. Every command takes its
ligand, residue names, and file paths as arguments; nothing about a
specific molecule is hardcoded anywhere in this tree. Generated results
from actually running the workflow belong under `outputs/`, which is
git-ignored: FP values, rankings, RDF data, convergence summaries,
selection results and rendered images are per-system scientific results,
never meant to be committed here.

## What the ligand workflow does

Given a ligand structure and topology, it:

1. builds a standalone ligand-in-water MD system and equilibrates it,
2. runs unbiased production MD, monitoring block-by-block whether the
   ligand's local hydration structure (water RDF shape, per-atom WaterFP
   fingerprint, and the FP-based atom ranking) has converged, stopping
   early once it has,
3. computes the WaterFP fingerprint of every ligand heavy atom over the
   converged block and, from it, picks an anti-bulk pair (G1) and a bulk
   pair (G2) with the official WaterFP selection rules,
4. builds a starting-point PLUMED fragment (`GROUP` + `COORDINATION`) for
   using G1/G2 as a ligand-side collective variable.

An optional 5th step renders the selected atoms on the bound complex for
a quick visual check. Why each step works as it does is in
[docs/METHOD_RATIONALE.md](docs/METHOD_RATIONALE.md).

## Package layout

```text
├── pyproject.toml              <- installable package (pip install -e .)
├── configs/example.yaml        <- settings for one system (oneopes --config ...)
├── docs/                       <- one page per stage, protein.md, funnel.md, METHOD_RATIONALE.md
├── tests/reference/            <- vendored upstream selection code (test reference)
└── src/oneopes/
    ├── system.py, waterfp.py, trajectory_blocks.py,
    │   convergence.py, block_run.py, reports.py, provenance.py  <- shared core
    ├── ligand/
    │   ├── setup.py, mdp/       <- 1. build the ligand-only MD system
    │   ├── monitor.py           <- 2. block-wise hydration convergence monitor
    │   ├── g1_g2.py             <- 3. WaterFP -> G1/G2
    │   └── plumed_cv.py         <- 4. starting-point PLUMED CV fragment
    ├── render/                  <- 5. optional: render selected atoms
    ├── protein/                 <- protein-side analysis (`oneopes protein`)
    ├── funnel/, vectors.py      <- BioSimSpace funnel CV
    └── cli.py                   <- the `oneopes` command, one subcommand per stage
```

Each subcommand is a typed function in its stage's module (e.g.
`oneopes.ligand.monitor.monitor`); `cli.py` registers it with
[cyclopts](https://cyclopts.readthedocs.io), which builds the flags from its
parameters and the help from its docstring. Options shared by several
commands (`SystemOptions`, `HydrationOptions`, `FrameWindow`, `BlockOptions`)
are dataclasses whose fields become flags.

## Install

```bash
pip install -e .        # or: uv pip install -e .
pip install -e ".[dev]" && pytest   # with the tests
```

This installs the `oneopes` package and the `oneopes` command;
`oneopes --help` lists its subcommands. Additional, non-pip dependencies:

- GROMACS (tested against 2021.7) on `PATH`, for stage 1/production MD
- PyMOL and/or UCSF Chimera, only if you use stage 5's rendering scripts
  (invoked by file path, not as subcommands - see [docs/render.md](docs/render.md))
- BioSimSpace (conda-only), only for `oneopes.funnel`

## Input requirements

- A ligand structure with a GROMACS-compatible topology (built however
  suits your project - CGenFF/CHARMM-GUI, GAFF/acpype, extraction from an
  existing complex's topology). The ligand and water are found
  automatically (`oneopes.system`); every analysis command accepts
  `--ligand-selection`/`--water-selection` to choose them explicitly.
- A 3-point water model consistent with the rest of your project.
- For stage 5 (optional): a bound-complex structure (protein + ligand) to
  render the selected atoms on.

## How to run the workflow

```bash
# Stage 1 - build the system
oneopes prepare-system \
    --ligand-gro ligand.gro --ligand-top ligand.top --outdir outputs/lig_system/

# ... production MD (see docs/ligand_setup.md), then:

# Stage 2 - convergence monitor (auto-stops production once converged).
# Stage 3 runs here too: on convergence it writes rdf.csv, fingerprints.csv
# and g1_g2.yaml for the converged block to outputs/convergence/selection.
oneopes monitor $MDRUN_PID \
    --tpr outputs/lig_system/prod.tpr --xtc outputs/lig_system/prod.xtc \
    --outdir outputs/convergence

# Stage 3 by hand: on another block of the monitor's run, or over any frame window
# oneopes select-block --tpr ... --run outputs/convergence/blocks.nc --block N --outdir outputs/selection
# oneopes select --tpr ... --xtc ... --first-frame N --last-frame M --outdir outputs/selection

# Stage 4 - ligand-side CV fragment
oneopes build-cv \
    --g1-g2 outputs/convergence/selection/g1_g2.yaml --out outputs/ligand_cv/plumed_fragment.dat

# Stage 5 - optional visualization, see docs/render.md
```

Instead of retyping flags, keep each system's settings in a YAML file with
one section per command (start from `configs/example.yaml`) and run
`oneopes --config my_system.yaml monitor`. A flag on the command line still
overrides the file, and a misspelt setting is an error.

Every analysis command also writes `run_metadata.json`: its input files'
names and SHA-256, its options and the package versions.

## Commands

| Stage | Command | Module | Purpose | Notes |
|---|---|---|---|---|
| 1 | `oneopes prepare-system` | `oneopes.ligand.setup` | Ligand-only system build (box, solvate, ionize, EM/NVT/NPT) | [ligand_setup.md](docs/ligand_setup.md) |
| 2 | `oneopes monitor` | `oneopes.ligand.monitor` | Block-wise hydration convergence monitor + auto-stop | [ligand_monitor.md](docs/ligand_monitor.md) |
| 3 | `oneopes select`, `oneopes select-block` | `oneopes.ligand.g1_g2` | G1/G2 by the official WaterFP selection rules, over a frame window or a block of the monitor's `blocks.nc` | [g1_g2.md](docs/g1_g2.md) |
| 4 | `oneopes build-cv` | `oneopes.ligand.plumed_cv` | Starting-point PLUMED CV fragment from G1/G2, to review by hand | [plumed_cv.md](docs/plumed_cv.md) |
| 5 | `oneopes add-legend` (+ PyMOL/Chimera scripts) | `oneopes.render` | Optional: render selected atoms on the bound complex | [render.md](docs/render.md) |
| - | `oneopes protein` | `oneopes.protein.analysis` | Protein-side WaterFP, contacts and SASA around the bound ligand | [protein.md](docs/protein.md) |
