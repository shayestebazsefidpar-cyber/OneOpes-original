# Stage 1 - ligand-in-water system

`oneopes prepare-system` builds and equilibrates the ligand alone in water:
box → solvate → ions → EM → NVT → NPT, with GROMACS (tested with 2021.7).

Needs: a ligand `.gro` and GROMACS topology (CGenFF, GAFF/acpype, or taken from a complex),
and the same 3-point water model as the rest of your project.

```bash
oneopes prepare-system --ligand-gro ligand.gro --ligand-top ligand.top --outdir outputs/lig_system
cd outputs/lig_system
gmx grompp -f prod.mdp -c npt.gro -t npt.cpt -p system.top -o prod.tpr
gmx mdrun -deffnm prod -v &
oneopes monitor $! --tpr prod.tpr --xtc prod.xtc
```

- The `.mdp` files come from `src/oneopes/ligand/mdp/` (`--mdp-dir` for your own);
  `prod.mdp` is copied to `--outdir`. Its 100 ns length is only a ceiling: the monitor
  usually stops the run much earlier.
- The box keeps `--box-distance-nm` (1.2 nm) between the ligand and its edge, and is at least
  `--min-box-nm` (4.2 nm): WaterFP counts waters up to 2.001 nm away, so a smaller box would count
  some twice through the periodic boundary. Raise it if you raise the monitor's `--rmax-nm`.
- `--start-from nvt_tpr` (or any step) resumes a partial build.
- grompp warnings stop the build, except the expected net-charge warning before adding ions.
  `--maxwarn N` lets EM/NVT/NPT through N warnings you have checked.
- Position restraints: make `posre_lig.itp` with `gmx genrestr` on the **ligand topology
  alone**. Run on a larger system, it gets global atom indices and restrains the wrong atoms.
- Stop mdrun with SIGTERM, never SIGKILL: killing a GPU run can break the CUDA driver for the
  whole machine until a kernel module is reloaded. The monitor uses SIGTERM.
