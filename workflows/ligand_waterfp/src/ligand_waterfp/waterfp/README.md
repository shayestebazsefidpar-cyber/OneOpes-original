# Stage 3 - WaterFP calculation

The core hydration math, independent of any convergence/blocking concept:
per-atom water radial density profile (RDF) and the WaterFP fingerprint
(FP) derived from it, exactly matching the reference implementation
(github.com/valeriorizzi/WaterFP, `Scripts/calc_rdf.sh` + `fp.py`).

- **`calculate_rdf.py`** (library): given solute/water AtomGroups + a
  frame range, computes each solute atom's raw water number-density
  profile n(r) (`compute_density_profile()`, `make_bins()`).
- **`calculate_fingerprint.py`** (library): given n(r), computes the
  scalar FP and g(r) per atom (`fp_from_density_profile()`,
  `fingerprints_from_rdf_table()`).
- **`cli.py`**: the `ligand-waterfp` command wrapping both, with
  subcommands `run` (trajectory -> rdf.csv + fingerprints.csv, the
  single-shot counterpart to
  `ligand_waterfp.convergence.monitor_convergence`), `rdf` (the
  expensive trajectory pass alone, cacheable) and `fingerprint`
  (re-fingerprint a cached rdf.csv, no trajectory needed).

`monitor_convergence.py` in Stage 2 imports these two modules directly
rather than duplicating the RDF/FP math - this folder is the single source
of truth for "what is a WaterFP fingerprint," used identically whether
you're checking convergence online or computing a final answer once.

## Usage

**Restrict `--start-frame`/`--end-frame` to Stage 2's converged block**
(its `convergence_summary.json`'s `stop_block` x `block_size_ns`), not the
whole trajectory - see `../convergence/README.md`'s "Next stage" section
for why (confirmed empirically: a whole-trajectory average changes which
atom the official algorithm's tie-break picks, even past convergence).

```bash
ligand-waterfp run --tpr prod.tpr --xtc prod.xtc \
    --start-frame <computed from stop_block> --end-frame <computed from stop_block> \
    --outdir outputs/fingerprints

# or the two steps separately
ligand-waterfp rdf --tpr prod.tpr --xtc prod.xtc --out outputs/rdf/rdf.csv
ligand-waterfp fingerprint --rdf-csv outputs/rdf/rdf.csv --out outputs/fingerprints/fingerprints.csv
```

`fingerprints.csv` (columns `atom,fp`) is the starting point for the
ranking CSV `ligand_waterfp.official_selection.run_official_selection`
expects (see that stage's README for the extra `name`/`atom`(serial)
columns the official algorithm needs).
