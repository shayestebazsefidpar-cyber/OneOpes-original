# Stage 2 - convergence monitor

`oneopes monitor` measures WaterFP for every ligand heavy atom block by block as mdrun writes
the trajectory, compares each block with the previous one (see `METHOD_RATIONALE.md`), and
on convergence stops mdrun and selects G1/G2 on the converged block.

```bash
oneopes monitor $MDRUN_PID --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence   # live
oneopes monitor --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence              # finished run
```

Settings: `--block-ns`, `--fp-rel-tol`, `--rdf-nrmsd-tol`, `--spearman-tol`, `--n-stable`,
`--ligand-selection`, ... (`oneopes monitor --help`).

Writes to `--outdir`:

- `blocks.nc`: the whole run as one xarray Dataset, rewritten after every block
  (`oneopes.block_run.load_run`);
- per-block CSVs of FP and g(r), `convergence_metrics.csv`, `convergence_summary.json`;
- `fp_vs_time.png` and `g_per_block.png`, one panel per ligand atom, when the run ends;
- `run_metadata.json` when the run ends;
- on convergence, `selection/` with `rdf.csv`, `fingerprints.csv` and `g1_g2.yaml`.

To select again, e.g. on another block or with another `--fp-round-decimals`, use the saved run:

```bash
oneopes select-block --tpr prod.tpr --run outputs/convergence/blocks.nc --outdir outputs/selection --block N
```

Select on one block, not the whole trajectory: on a real system a whole-trajectory FP kept the
G1/G2 anchors but changed the G1 partner.
