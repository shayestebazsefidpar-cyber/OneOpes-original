# Stage 1 - ligand-in-water system

`oneopes prepare-system` builds and equilibrates the ligand alone in water:
box → solvate → ions → EM → NVT → NPT, with GROMACS (tested with 2021.7).

Needs: a ligand `.gro` and GROMACS topology (CGenFF, GAFF/acpype, or taken from a complex),
and the same 3-point water model as the rest of your project.

```bash
oneopes prepare-system --ligand-gro ligand.gro --ligand-top ligand.top --outdir outputs/lig_system
cd outputs/lig_system
gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p system.top -o prod.tpr -maxwarn 1
gmx mdrun -deffnm prod -v &
oneopes monitor $! --tpr prod.tpr --xtc prod.xtc
```

- The `.mdp` files come from `src/oneopes/ligand/mdp/` (`--mdp-dir` for your own);
  `prod.mdp` is copied to `--outdir`. Its 100 ns length is only a ceiling: the monitor
  usually stops the run much earlier.
- `--start-from nvt_tpr` (or any step) resumes a partial build.
- Position restraints: make `posre_lig.itp` with `gmx genrestr` on the **ligand topology
  alone**. Run on a larger system, it gets global atom indices and restrains the wrong atoms.
- Stop mdrun with SIGTERM, never SIGKILL: killing a GPU run can break the CUDA driver for the
  whole machine until a kernel module is reloaded. The monitor uses SIGTERM.
