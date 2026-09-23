# Stage 2 - Unbiased MD convergence monitoring

Watches an in-progress (or already-finished) ligand-only production
trajectory and decides, block by block, whether the hydration data is
converged enough to stop early or to trust for the next stage.

- **What it does**: splits the trajectory into consecutive blocks, computes
  each ligand heavy atom's water RDF + WaterFP fingerprint per block (via
  `ligand_waterfp.waterfp.calculate_rdf` and
  `ligand_waterfp.waterfp.calculate_fingerprint`), and
  checks three stability criteria between consecutive blocks. See
  `METHOD_RATIONALE.md` for the full reasoning and formulas.
- **Script**: `monitor_convergence.py`
- **Inputs**: a `.tpr` + `.xtc` (can be a still-growing trajectory from a
  live `mdrun`, or a finished one for a one-shot report)
- **Outputs** (written to `--outdir`, default `outputs/convergence` -
  git-ignored): per-block FP/RDF CSVs, a convergence-metrics CSV, plots,
  and `convergence_summary.json`/`.txt`

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

Once `convergence_summary.json` reports `"converged": true`, it also
reports `"stop_block"` and `"block_size_ns"`. **Use those to compute the
exact frame range of that specific converged block** - not the whole
trajectory - when calling `ligand_waterfp.waterfp.run_waterfp` next:

```text
frames_per_block = round(block_size_ns * 1000 / dt_ps)   # dt_ps from monitor_convergence's own log line
start_frame = stop_block * frames_per_block
end_frame   = (stop_block + 1) * frames_per_block
```

This matters: the official selection algorithm's 0.2 fp_round tie-break
(`official_selection`) is sensitive to exactly which frames the FP values
were averaged over. Confirmed empirically against a real system's data -
a whole-trajectory average reproduced the correct G1/G2 *anchor* atoms but
picked a different G1 *partner* atom than using the converged block alone;
using the exact converged-block frame range reproduced the historical
result exactly. A whole-trajectory average is not equivalent, even once
the run has converged.
