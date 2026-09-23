# Stage 1 - Ligand-only system setup

Builds a standalone ligand-in-water GROMACS system (no protein present) as
the starting point for the unbiased MD run that Stage 2 monitors:
box definition -> solvation -> ionization -> EM -> NVT -> NPT.

`prepare_ligand_system.py` wraps this whole sequence into one reproducible
command. Everything it *produces* (`.gro`/`.top`/`.tpr`/`.cpt`/`.itp`
files) is per-system output, not code - run it into `--outdir` under your
own `outputs/` tree (git-ignored), never into this folder.

## Prerequisites

- A ligand structure with a GROMACS-compatible topology (e.g. from
  CGenFF/CHARMM-GUI, GAFF/acpype, or extracted from an existing
  protein-ligand system's own topology). This workflow assumes the ligand
  residue is named `MOL` in the resulting topology by default - every
  script downstream takes `--ligand-resname` if yours differs.
- A TIP3P (or compatible 3-point) water model, matching whatever water
  model the rest of your project uses elsewhere, so the ligand-only
  hydration reference stays consistent with any bound-complex simulations
  it will be compared against.
- GROMACS on your `PATH` (this workflow was built against 2021.7).

## Usage

```bash
ligand-waterfp-prepare-system \
    --ligand-gro ligand.gro --ligand-top ligand.top \
    --outdir outputs/lig_system/ \
    --mdp-dir mdp/                # defaults to this folder's mdp/, override if needed
```

Then run production yourself (so you control how/where mdrun is launched
and can capture its PID):

```bash
cd outputs/lig_system/
gmx grompp -f ../../mdp/prod.mdp -c npt.gro -t npt.cpt -p system.top -o prod.tpr -maxwarn 1
gmx mdrun -deffnm prod -v &
python ../`ligand_waterfp.convergence.monitor_convergence` $! --tpr prod.tpr --xtc prod.xtc
```

`prod.mdp` sets a 100 ns cap as a safety ceiling only - in practice
Stage 2 routinely stops production far earlier once hydration has
converged (see `convergence/METHOD_RATIONALE.md`).

Resume a partially-completed build with `--start-from`, e.g.
`--start-from nvt_tpr` if `em` already finished.

## A note on position restraints

If your ligand topology uses `#ifdef POSRES` / `#include "posre_lig.itp"`
during EM/NVT/NPT, generate `posre_lig.itp` with `gmx genrestr` on the
**isolated ligand topology**, not on a larger system that includes it as a
sub-molecule. Restraint files included inside a molecule's own
`moleculetype` block need atom indices local to that molecule - running
`genrestr` against a bigger system and reusing the result produces global
indices that silently restrain the wrong atoms. `prepare_ligand_system.py`
does not generate restraint files itself; add this step yourself before
`em_tpr` if your topology needs it.

## Never SIGKILL a GPU-active production run

If you stop `prod` mid-run (e.g. once Stage 2 signals convergence, or
manually), always send `SIGTERM` and wait for GROMACS to exit on its own.
Forcibly killing a GPU-active `mdrun` with `SIGKILL` has been confirmed to
desync the CUDA runtime from the NVIDIA driver for the whole machine, not
just that process - recovering requires reloading a kernel module as
root. `SIGTERM` lets GROMACS release its CUDA context cleanly and avoids
this entirely. Stage 2's monitor already does this correctly when it
auto-stops a converged run.
