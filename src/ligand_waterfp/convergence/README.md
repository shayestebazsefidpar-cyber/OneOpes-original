# Stage 2 - Unbiased MD convergence monitoring

Watches an in-progress (or already-finished) ligand-only production
trajectory and decides, block by block, whether the hydration data is
converged enough to stop early or to trust for the next stage.

- **What it does**: splits the trajectory into consecutive blocks, computes
  each ligand heavy atom's water RDF + WaterFP fingerprint per block (via
  the shared `waterfp.WaterFP` analysis), and
  checks three stability criteria between consecutive blocks. See
  `METHOD_RATIONALE.md` for the full reasoning and formulas.
- **Script**: `monitor_convergence.py`
- **Inputs**: a `.tpr` + `.xtc` (can be a still-growing trajectory from a
  live `mdrun`, or a finished one for a one-shot report)
- **Outputs** (written to `--outdir`, default `outputs/convergence` -
  git-ignored): per-block FP/RDF CSVs, a convergence-metrics CSV, plots,
  `convergence_summary.json`, and on convergence the G1/G2
  selection in `selection/`

## Usage

```bash
# live monitoring, auto-SIGTERM the mdrun once converged
gmx mdrun -deffnm prod -v &
ligand-waterfp-monitor $! --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence

# one-shot report on an already-finished trajectory (no PID -> no auto-stop)
ligand-waterfp-monitor --tpr prod.tpr --xtc prod.xtc --outdir outputs/convergence
```

Override any threshold/setting via CLI flags (`--block-ns`, `--fp-rel-tol`,
`--rdf-nrmsd-tol`, `--spearman-tol`, `--n-stable`, `--ligand-resname`,
`--water-resname`, `--water-atom-name`, ...) - see `ligand-waterfp-monitor
--help`, or set them once in `../../../configs/ligand_waterfp.yaml` and
pass them through your own driver.

## Next stage

On convergence the monitor also runs the G1/G2 selection on the converged
block and writes `rdf.csv`, `fingerprints.csv` and `g1_g2.yaml` to
`--outdir/selection` (`--fp-round-decimals` sets the FP rounding). Pass
that `g1_g2.yaml` to `ligand-waterfp-build-cv`.

To redo the selection by hand, run `ligand-waterfp-select` over the same
block (`converged_block` and `block_ns` are in `convergence_summary.json`),
not the whole trajectory:

```text
frames_per_block = round(block_ns * 1000 / dt_ps)   # dt_ps from monitor_convergence's own log line
start_frame = converged_block * frames_per_block
end_frame   = (converged_block + 1) * frames_per_block
```

The block matters: the G1/G2 selection's 0.2 fp_round tie-break
(`g1_g2_selection`) is sensitive to exactly which frames the FP values
were averaged over. Confirmed empirically against a real system's data -
a whole-trajectory average reproduced the correct G1/G2 *anchor* atoms but
picked a different G1 *partner* atom than using the converged block alone;
using the exact converged-block frame range reproduced the historical
result exactly. A whole-trajectory average is not equivalent, even once
the run has converged.
