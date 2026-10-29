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

This is **code only** - no example systems, no HSP90/BRD4 results, no
trajectories, no committed scientific output. Every script takes its
ligand, residue names, and file paths as arguments; nothing about a
specific molecule is hardcoded anywhere in this tree. Generated results
from actually running the workflow belong under `outputs/`, which is
git-ignored (see `.gitignore`).

## 1. What this workflow does

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
a quick visual check.

## 2. Package layout

```text
├── pyproject.toml              <- installable package (pip install -e .)
├── configs/example.yaml        <- settings for one system (oneopes --config ...)
├── docs/                        <- per-stage notes and METHOD_RATIONALE.md
├── tests/reference/             <- vendored upstream selection code (test reference)
└── src/oneopes/
    ├── waterfp.py, convergence.py, trajectory_blocks.py, system.py  <- shared core
    ├── ligand/
    │   ├── setup.py, mdp/       <- 1. build the ligand-only MD system
    │   ├── monitor.py           <- 2. block-wise hydration convergence monitor
    │   ├── g1_g2.py             <- 3. WaterFP over the converged block -> G1/G2
    │   └── plumed_cv.py         <- 4. starting-point PLUMED CV fragment
    ├── render/                  <- 5. optional: render selected atoms
    ├── protein/                 <- protein-side analysis (`oneopes protein`)
    ├── funnel/                  <- BioSimSpace funnel CV
    └── cli.py                   <- the `oneopes` command, one subcommand per stage
```

Each stage has its notes in `docs/`. Each subcommand is a typed function in
its stage's module (e.g. `oneopes.ligand.monitor.monitor`); `cli.py`
registers it with [cyclopts](https://cyclopts.readthedocs.io), which builds
the flags from its parameters and the help from its docstring. Shared
settings such as `Tolerances` and `RadialGrid` are dataclasses whose fields
become flags.

## 3. Install

```bash
pip install -e .        # or: uv pip install -e .
```

This installs the `oneopes` package and the `oneopes` command;
`oneopes --help` lists its subcommands, one per stage (see section 6).

Additional, non-pip dependencies:

- GROMACS (tested against 2021.7) on `PATH`, for stage 1/production MD
- PyMOL and/or UCSF Chimera, only if you use stage 5's rendering scripts
  (invoked by file path, not as subcommands - see `docs/render.md`)

## 4. Input requirements

- A ligand structure with a GROMACS-compatible topology (built however
  suits your project - CGenFF/CHARMM-GUI, GAFF/acpype, extraction from an
  existing complex's topology). The ligand and water are found
  automatically (`oneopes.system`); every analysis command accepts
  `--ligand-selection`/`--water-selection` to choose them explicitly.
- A 3-point water model consistent with the rest of your project.
- For stage 5 (optional): a bound-complex structure (protein + ligand) to
  render the selected atoms on.

## 5. How to run the workflow

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

# Stage 3 by hand, only to select over another frame window (see
# docs/ligand_monitor.md for the converged block's frame range):
# oneopes select --tpr ... --xtc ... --start-frame N --end-frame M --outdir outputs/selection

# Stage 4 - ligand-side CV fragment
oneopes build-cv \
    --g1-g2 outputs/convergence/selection/g1_g2.yaml --out outputs/ligand_cv/plumed_fragment.dat

# Stage 5 - optional visualization, see docs/render.md
```

Instead of retyping flags, keep each system's settings in a YAML file with
one section per command (start from `configs/example.yaml`) and run
`oneopes --config my_system.yaml monitor`. A flag on the command line still
overrides the file, and a misspelt setting is an error.

## 6. What each stage does

| Stage | Module | Command | Purpose |
|---|---|---|---|
| 1 | `oneopes.ligand.setup` | `oneopes prepare-system` | Ligand-only system build (box, solvate, ionize, EM/NVT/NPT) |
| 2 | `oneopes.ligand.monitor` | `oneopes monitor` | Block-wise hydration convergence monitor + auto-stop |
| 3 | `oneopes.ligand.g1_g2` | `oneopes select` | WaterFP over the converged block, then G1/G2 by the official selection rules |
| 4 | `oneopes.ligand.plumed_cv` | `oneopes build-cv` | Builds a starting-point PLUMED CV fragment from G1/G2 |
| 5 | `oneopes.render` | `oneopes add-legend` (+ PyMOL/Chimera scripts by path) | Optional: render selected atoms on the bound complex |
| - | `oneopes.protein.analysis` | `oneopes protein` | Protein-side WaterFP, contacts and SASA around the bound ligand |

## 7. Where generated outputs are written

Every script writes under an `outputs/` (or user-specified) directory that
is git-ignored by `.gitignore` in this folder. Nothing under `outputs/` -
FP values, rankings, RDF data, convergence summaries, selection results,
rendered images - is ever meant to be committed here; those are per-system
scientific results, not workflow code.

## 8. How the official WaterFP implementation is used

The selection rules come from the upstream notebook
`Scripts/fp_driven_atom_selection.ipynb` of
`https://github.com/valeriorizzi/WaterFP` (commit
`6a5aef725a66b28e86d3582aee3e9e8bbbe4bc54`). Its selection functions are
vendored verbatim in
`tests/reference/vendored_selection.py`, and
`tests/test_g1_g2_reference.py` checks that `select_g1_g2` picks the same
atoms on hundreds of random ligands.

## 9. How G1/G2 are generated

G1 (the "anti-bulk" pair) is the ligand heavy atom with the most negative
WaterFP plus a partner among the next most negative atoms that is not
bonded to it, preferring the farthest when FP values are close. G2 (the
"bulk" pair) is the atom with the least negative WaterFP plus the next
least negative atom at least three bonds away. See
`oneopes.ligand.g1_g2`.

## 10. How the resulting ligand CV is constructed

Stage 4 takes G1/G2's atom serials and emits a PLUMED fragment defining
each as a `GROUP`, plus a `COORDINATION` CV for each against a
water-oxygen reference group. This is a starting point requiring manual
review (switching-function parameters, the water-reference group, and
merging into your system's real `plumed.dat`) - see
`docs/plumed_cv.md`.

# oneopes.funnel

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
from oneopes.funnel.geometry import funnel_extent, funnel_wall_points
from oneopes.funnel.make_funnel import make_funnel

system = BSS.IO.readMolecules(["top.top", "npt.gro"])
f = make_funnel(system)                       # extra kwargs go to BSS Funnel(...)
f.p0p1.atoms0, f.p0p1.atoms1                  # BSS makeFunnel selections
f.p0p1.p0, f.p0p1.p1, f.p0p1.axis             # derived from those selections (A)
f.parameters                                  # width, buffer, steepness, inflection, bounds (nm)
f.cv                                          # the BioSimSpace Funnel object (e.g. f.cv.getCorrection())
funnel_extent(f.cv, [0.5, 1.0, 2.0])          # radius (nm) via Funnel.getExtent()
funnel_wall_points(f.p0p1, f.cv, step_A=2.0)  # (N, 3) wall points in A
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
| `bss.py` | the BioSimSpace calls, with lengths and positions as plain numbers |
| `p0_p1.py` | `makeFunnel()` selections + derived P0/P1 |
| `geometry.py` | CV parameters, `getExtent()` radius, wall points |
| `make_funnel.py` | `make_funnel()` - selections + the BioSimSpace `Funnel` CV |
| `plumed_io.py` | (stub) read-only parsing of existing `plumed.dat` files |
| `system_io.py` | (stub) system loading |

Code in `src/oneopes/funnel/`, tests in `tests/funnel/`.

# oneopes.protein (`oneopes protein`)

Protein-side WaterFP hydration analysis of one protein-ligand system, run
**retrospectively on an existing, completed unbiased MD trajectory**. The
trajectory length is read from the input (any length); the whole trajectory is
analysed and nothing is stopped when convergence is reached. It is
standalone and system-independent: every input is passed explicitly, and the
code contains no system-specific atoms, coordinates, walls or boundaries.

## Pipeline

```text
complex structure ──► protein heavy atoms within --candidate-cutoff-A (default 7 Å) of the ligand
                      (fixed candidate pool)
trajectory (length detected from the input)
           ──► consecutive --block-ns (default 5 ns) blocks over the whole trajectory
               per block and candidate: WaterFP / RDF, Contact (5 Å), Bridge (3.5 Å water)
           ──► metrics for every block transition ──► first sustained convergence
               (analysis continues to the end of the trajectory)
           ──► full-trajectory FP / Contact / Bridge + static SASA
           ──► final_per_atom_table.csv
```

* **Candidates**: protein (MDAnalysis `protein`) heavy atoms (topology mass
  ≥ 1.5, not H) whose minimum distance to any ligand heavy atom in
  `--complex` is ≤ the cutoff. Computed once; never filtered afterwards.
* **WaterFP / RDF**: the shared `oneopes.waterfp.WaterFP` analysis: water-oxygen number
  density 0–2.001 nm in 0.001 nm bins, bulk norm from the last 500 bins.
* **Contact**: fraction of frames with the candidate within 5 Å of any ligand
  heavy atom. **Bridge**: fraction of frames with ≥1 water oxygen within
  3.5 Å of both the candidate and a ligand heavy atom.
* **SASA**: static, on the complex (protein + ligand heavy atoms, no
  water/ions), freesasa defaults.
* **Convergence**: the shared `oneopes.convergence.check_stability` with FP relative
  change ≤ 0.10, RDF nRMSD ≤ 0.15, Spearman ≥ 0.90 — all three in a
  transition (FP change and nRMSD are the worst case over all candidates).
  Sustained convergence = 3 consecutive stable transitions; the end time of
  the block completing the third one is `converged_ns`.
  This is only reported: every block up to the end of the trajectory is still
  analysed, so later loss of stability is visible in `convergence_metrics.csv`.
* **Times**: block *k* spans `t(first frame of k)` to that + block size;
  the trajectory runs from the time of its first frame to that of its last. Frames
  after the last complete block are not in any block but are included in the
  final values.

## Interpretation and limitations

* **What FP measures.** FP quantifies how much the local water-oxygen
  distribution around an atom deviates from bulk water. With the current
  WaterFP definition the integrand `-2π·norm·(g ln g − g + 1)·r²` is never
  positive, so FP ≤ 0 (FP = 0 for a bulk-like distribution).
* **More negative FP does not mean "more hydrated".** Buried or dry protein
  atoms can have strongly negative FP because the protein excludes water and
  strongly perturbs the distribution around them. `FP_Rank` 1 is the most
  negative FP, i.e. the strongest perturbation, not automatically the most
  hydrated atom. Choosing final hydration CVs requires further scientific
  interpretation (e.g. together with `Contact_occupancy_5A`,
  `Bridge_occupancy` and `SASA_A2`); this workflow does not select CVs.
* **Convergence criterion.** `fp_change` is a worst-case per-atom
  relative criterion, `max |FP_now − FP_prev| / max(|FP_prev|, 1e-6)` over
  all candidates. Atoms with FP close to zero (bulk-like, solvent-exposed) can
  produce very large relative changes and can dominate the verdict. This is
  inherited from the ligand-side criterion (now the shared
  `oneopes.convergence.check_stability`) and is intentionally unchanged here.
* **Reference structure.** Candidate selection and `SASA_A2` use only the
  supplied `--complex` structure. SASA is static (one structure), not
  trajectory-averaged.
* **Distance definition.** `Min_distance_to_ligand_A` is the minimum distance
  from the protein atom to any ligand heavy atom, not to the ligand centre of
  mass.
* **Numbering.** `Atom_name` and `Residue_number` come from the topology
  universe (`--topology`); `Chain_ID` comes from the raw `--complex` file when
  it has chain IDs. Residue numbering can therefore differ between a `.tpr`
  and the PDB. `Atom_serial` is the atom's 1-based number in the topology
  (`Atom_index` + 1), as GROMACS and PLUMED count atoms; it is not the serial
  column of the PDB file, which can differ (e.g. after TER records).
* **Scope.** The workflow is retrospective post-processing of an existing
  trajectory. It does not run MD, OneOPES or PLUMED, and has no Zone or
  funnel logic.

## Usage

```bash
oneopes protein --complex complex.pdb --topology prod.tpr --trajectory prod.xtc \
    --output-dir outputs/run1
```

Options: `--candidate-cutoff-A`, `--first-frame`, `--last-frame` (exclusive),
`--ligand-selection`, `--water-selection`, and the same `--block-ns` and
tolerance flags as `oneopes monitor` (`--fp-rel-tol`, `--rdf-nrmsd-tol`,
`--spearman-tol`, `--n-stable`).

The topology must carry masses and match the trajectory atom-for-atom. The
complex must have the same atom count and the same protein/ligand heavy-atom
names (water/hydrogen naming may differ between formats).

## Outputs (all in `--output-dir`)

| File | Content |
|---|---|
| `candidate_atom_mapping.csv` | the candidate pool |
| `fp_values_per_block.csv`, `rdf_profiles_per_block.csv` | FP (with the block times) and g(r) per block and atom |
| `contact_occupancy_per_block.csv` | Contact / Bridge per block and atom |
| `fp_values_by_block_wide.csv`, `fp_ranking_by_block_wide.csv` | FP and rank (1 = most negative) per block |
| `convergence_metrics.csv` | one row per block after the first, compared with the block before: `block`, `fp_change`, `rdf_nrmsd`, `spearman_rho`, `fp_pass`, `rdf_pass`, `spearman_pass`, `passed`, `stable_in_a_row`, `t_start_ns`, `t_end_ns`, `n_frames`, `t_mid_ns` |
| `convergence_summary.json` | block size, tolerances, number of blocks, `converged_block`/`converged_ns`, unstable transitions after convergence, trajectory and analysed time range |
| `final_per_atom_table.csv` | one row per candidate, sorted by `FP_Rank` |
| `final_per_atom_table.md` | the same table as readable Markdown (floats rounded; the CSV keeps full precision) |
| `run_metadata.json` | inputs (file names, sha256), options, frames/blocks, versions |

Final table columns: `FP_Rank, Atom, Atom_index, Atom_serial, Atom_name,
Residue_name, Residue_number, Chain_ID, Min_distance_to_ligand_A, FP,
Contact_occupancy_5A, Bridge_occupancy, SASA_A2`. FP, Contact and Bridge are
computed over the full analysed frame range with the same code as the blocks;
SASA is static; `FP_Rank` 1 = most negative FP. Block-wise metrics are only in
`convergence_metrics.csv`; the analysed time range and the time of first
sustained convergence (`converged_ns`) are in `convergence_summary.json`.

## Install and test

```bash
pip install -e ".[dev]"   # or: uv pip install -e ".[dev]"
pytest
```
