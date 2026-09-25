# ligand_waterfp

A reusable, system-independent, **installable Python package** for
ligand-side hydration-CV construction:

```text
ligand-only unbiased MD
        |
WaterFP calculation
        |
block-wise convergence
        |
official WaterFP atom selection
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
3. computes the final WaterFP fingerprint for every ligand heavy atom,
4. runs the **official**, unmodified WaterFP atom-selection algorithm
   (vendored from the upstream repository) on the converged FP ranking to
   pick an anti-bulk pair (G1) and a bulk pair (G2),
5. formalizes that result as structured data,
6. builds a starting-point PLUMED fragment (`GROUP` + `COORDINATION`) for
   using G1/G2 as a ligand-side collective variable.

An optional 7th step renders the selected atoms on the bound complex for
a quick visual check.

## 2. Package layout

```text
ligand_waterfp/
├── pyproject.toml              <- installable package (pip install -e .)
├── configs/ligand_waterfp.yaml <- generic parameter template
├── workflow.yaml                <- machine-readable stage manifest
└── src/ligand_waterfp/
    ├── system_setup/            <- 1. build the ligand-only MD system
    ├── convergence/              <- 2. block-wise hydration convergence monitor
    ├── waterfp/                  <- 3. RDF + WaterFP fingerprint calculation
    ├── official_selection/       <- 4+5. vendored upstream selection algorithm,
    │                                 writes structured G1/G2 YAML directly
    ├── g1_g2_selection/           <- helper module used by official_selection
    │                                 (parse/write functions only, no CLI)
    ├── ligand_cv/                 <- 6. starting-point PLUMED CV fragment
    └── visualization/             <- 7. optional: render selected atoms
```

Each subpackage has its own `README.md` with full detail. `main()` in
every stage's driver script is registered as a console-script entry point
in `pyproject.toml` (see section 4).

## 3. Install

```bash
cd ligand_waterfp/
pip install -e .        # or: uv pip install -e .
```

This installs the `ligand_waterfp` package and ten `ligand-waterfp-*`
console commands (one per stage script - see section 4). `requirements.txt`
is provided as a fallback if you'd rather `pip install -r requirements.txt`
and run scripts via `python -m ligand_waterfp.<subpackage>.<module>`
without installing the package itself.

Additional, non-pip dependencies:

- GROMACS (tested against 2021.7) on `PATH`, for stage 1/production MD
- PyMOL and/or UCSF Chimera, only if you use stage 7's rendering scripts
  (invoked by file path, not as console scripts - see
  `src/ligand_waterfp/visualization/README.md`)

## 4. Input requirements

- A ligand structure with a GROMACS-compatible topology (built however
  suits your project - CGenFF/CHARMM-GUI, GAFF/acpype, extraction from an
  existing complex's topology). The default assumed ligand residue name
  is `MOL`; every script accepts `--ligand-resname` if yours differs.
- A 3-point water model consistent with the rest of your project.
- For stage 7 (optional): a bound-complex structure (protein + ligand) to
  render the selected atoms on.

## 5. How to run the workflow

```bash
# Stage 1 - build the system
ligand-waterfp-prepare-system \
    --ligand-gro ligand.gro --ligand-top ligand.top --outdir outputs/lig_system/

# ... production MD (see src/ligand_waterfp/system_setup/README.md), then:

# Stage 2 - convergence monitor (auto-stops production once converged)
ligand-waterfp-monitor $MDRUN_PID \
    --tpr outputs/lig_system/prod.tpr --xtc outputs/lig_system/prod.xtc \
    --outdir outputs/convergence

# Stage 3 - final WaterFP fingerprints. IMPORTANT: restrict --start-frame/
# --end-frame to the SAME converged block Stage 2 reported (its
# convergence_summary.json's "stop_block" x block_size_ns / frame spacing),
# not the whole trajectory - the official selection's 0.2 fp_round
# tie-break is sensitive to exactly which window the FP values come from.
# Confirmed empirically: reproducing a real system's G1 partner atom
# only worked using the converged block's own frame range, not a
# whole-trajectory average (see convergence's own README for how to read
# stop_block/block_size_ns back into a frame range).
ligand-waterfp run \
    --tpr outputs/lig_system/prod.tpr --xtc outputs/lig_system/prod.xtc \
    --start-frame <stop_block * frames_per_block> --end-frame <(stop_block+1) * frames_per_block> \
    --outdir outputs/fingerprints

# Stage 4 - official atom selection (see its README: --system-id needs a
# hyphen, and load_mol() needs '<id, hyphen-parts joined>.tpr' in the CWD)
ligand-waterfp-prepare-ranking-csv \
    --tpr outputs/lig_system/prod.tpr --fp-csv outputs/fingerprints/fingerprints.csv \
    --system-id myproject-ligandA --out outputs/selection/ranking_input.csv

ln -s outputs/lig_system/prod.tpr myprojectligandA.tpr
# Stage 4+5 - official selection, parsed and written as structured G1/G2
# YAML directly in the same process (no intermediate text file anymore)
ligand-waterfp-select \
    --ranking-csv outputs/selection/ranking_input.csv \
    --system-id myproject-ligandA --out outputs/selection/g1_g2.yaml

# Stage 6 - ligand-side CV fragment
ligand-waterfp-build-cv \
    --g1-g2 outputs/selection/g1_g2.yaml --out outputs/ligand_cv/plumed_fragment.dat

# Stage 7 - optional visualization, see src/ligand_waterfp/visualization/README.md
```

`configs/ligand_waterfp.yaml` is a generic parameter template (paths and
thresholds, no results) you can copy per-system instead of retyping every
flag.

## 6. What each stage does

| Stage | Subpackage | Console command(s) | Purpose |
|---|---|---|---|
| 1 | `system_setup/` | `ligand-waterfp-prepare-system` | Ligand-only system build (box, solvate, ionize, EM/NVT/NPT) |
| 2 | `convergence/` | `ligand-waterfp-monitor` | Block-wise hydration convergence monitor + auto-stop |
| 3 | `waterfp/` | `ligand-waterfp` (`run`/`rdf`/`fingerprint` subcommands) | RDF + WaterFP fingerprint calculation (importable + standalone) |
| 4+5 | `official_selection/` (+ `g1_g2_selection/` helper) | `ligand-waterfp-prepare-ranking-csv`, `-select` | Official upstream selection algorithm (vendored, verbatim), parsed and written as structured G1/G2 YAML in the same process - see PR1 note below |
| 6 | `ligand_cv/` | `ligand-waterfp-build-cv` | Builds a starting-point PLUMED CV fragment from G1/G2 |
| 7 | `visualization/` | `ligand-waterfp-add-legend` (+ PyMOL/Chimera scripts by path) | Optional: render selected atoms on the bound complex |

## 7. Where generated outputs are written

Every script writes under an `outputs/` (or user-specified) directory that
is git-ignored by `.gitignore` in this folder. Nothing under `outputs/` -
FP values, rankings, RDF data, convergence summaries, selection results,
rendered images - is ever meant to be committed here; those are per-system
scientific results, not workflow code.

## 8. How the official WaterFP implementation is used

Stage 4's atom-selection logic is never reimplemented. The five functions
it needs - `load_mol()`, `path_length()`, `ranking()`, `select_next_atom()`,
`select_bulk_atom()` - are vendored verbatim (same logic, signatures,
variable names and control flow; only whitespace/PEP8 formatting was
normalized when copying out of the notebook's JSON source, confirmed by
diffing directly against it) from `https://github.com/valeriorizzi/WaterFP`
at commit `6a5aef725a66b28e86d3582aee3e9e8bbbe4bc54` - see
`src/ligand_waterfp/official_selection/WaterFP_official_scripts/README.md`
for the exact provenance and
`src/ligand_waterfp/official_selection/run_official_selection.py` for the
marked verbatim block. Only `main()`, below that block, was written for
this workflow.

## 9. How G1/G2 are generated

G1 (the "anti-bulk" pair) is the ligand heavy atom with the strongest
WaterFP fingerprint plus a partner atom chosen by the official algorithm's
graph-distance / FP-grouping tie-break rule (`select_next_atom`). G2 (the
"bulk" pair) is the same procedure applied from the opposite end of the FP
ranking (`select_bulk_atom`), representing the most solvent-like region of
the ligand. Both come directly out of the official algorithm; the only
work done afterward is parsing its two result sentences into the
structured YAML shape (`ligand_waterfp.g1_g2_selection.select_g1_g2`,
called in-process by `run_official_selection`'s own `main()` - no
recomputation, no intermediate file).

## 10. How the resulting ligand CV is constructed

Stage 6 takes G1/G2's atom serials and emits a PLUMED fragment defining
each as a `GROUP`, plus a `COORDINATION` CV for each against a
water-oxygen reference group. This is a starting point requiring manual
review (switching-function parameters, the water-reference group, and
merging into your system's real `plumed.dat`) - see
`src/ligand_waterfp/ligand_cv/README.md`.
