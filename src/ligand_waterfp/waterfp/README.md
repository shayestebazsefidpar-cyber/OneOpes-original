# Stage 3 - WaterFP calculation

The hydration math itself - per-atom water radial density profile (RDF) and
the WaterFP fingerprint (FP) derived from it, matching the reference
implementation (github.com/valeriorizzi/WaterFP, `Scripts/calc_rdf.sh` +
`fp.py`) - lives in the shared top-level `waterfp` module (`WaterFP`, an
MDAnalysis analysis class). It is used identically by this stage, by
Stage 2's `monitor_convergence.py` (once per block), and by `protein_waterfp`.

- **`run_waterfp.py`**: single-shot driver over a given frame range of a
  trajectory - the counterpart to
  `ligand_waterfp.convergence.monitor_convergence`, which runs the same
  `WaterFP` analysis repeatedly per block instead.

## Usage

**Restrict `--start-frame`/`--end-frame` to Stage 2's converged block**
(its `convergence_summary.json`'s `stop_block` x `block_size_ns`), not the
whole trajectory - see `../convergence/README.md`'s "Next stage" section
for why (confirmed empirically: a whole-trajectory average changes which
atom the official algorithm's tie-break picks, even past convergence).

```bash
ligand-waterfp-run-waterfp --tpr prod.tpr --xtc prod.xtc \
    --start-frame <computed from stop_block> --end-frame <computed from stop_block> \
    --outdir outputs/fingerprints
```

This writes `rdf.csv` (columns `atom,r_nm,n_r`, the water-O number density)
and `fingerprints.csv`.

`fingerprints.csv` (columns `atom,fp`) is the starting point for the
ranking CSV `ligand_waterfp.official_selection.run_official_selection`
expects (see that stage's README for the extra `name`/`atom`(serial)
columns the official algorithm needs).
