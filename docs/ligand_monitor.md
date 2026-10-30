# Stage 2 - convergence monitor

`oneopes monitor` measures WaterFP for every ligand heavy atom block by block as mdrun writes
the trajectory. After each block it checks where production starts, the FP of the production
blocks with its error, drift, and whether the G1/G2 selection is settled within those errors
(see `METHOD_RATIONALE.md`). On convergence it stops mdrun and selects G1/G2 on the production FP.

```bash
oneopes monitor $MDRUN_PID --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence   # live
oneopes monitor --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence              # finished run
```

Settings: `--block-ns` (1 ns), `--min-blocks` (10), `--drift-alpha` (0.05), `--n-passes` (2),
`--selection-confidence` (0.95), `--ligand-selection`, ... (`oneopes monitor --help`).
With the defaults a run takes at least 11 ns: ten production blocks and a second passing check.

Writes to `--outdir`:

- `blocks.nc`: the whole run as one xarray Dataset, rewritten after every block
  (`oneopes.block_run.load_run`);
- per-block CSVs of FP and g(r), `convergence_metrics.csv` (the check after each block),
  `convergence_summary.json`;
- `fp_vs_time.png` and `g_per_block.png`, one panel per ligand atom, when the run ends;
- `run_metadata.json` when the run ends;
- on convergence, `selection/` with `rdf.csv` (production density), `fingerprints.csv`
  (`fp`, its `fp_se`, `fp_uncorrected` and `drift_z`) and `g1_g2.yaml`.

To select again, e.g. on the production another check found or with another
`--fp-round-decimals`, use the saved run:

```bash
oneopes select-block --tpr prod.tpr --run outputs/convergence/blocks.nc --outdir outputs/selection --block N
```

Select on production, not the whole trajectory: on a real system a whole-trajectory FP kept the
G1/G2 anchors but changed the G1 partner.
