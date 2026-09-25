# Tests

Automated unit/regression tests, using synthetic data only - no real
system, ligand, or scientific result appears here.

Run with:
```bash
pip install -e ".[dev]"
pytest
```

## What's covered here

- `test_calculate_fingerprint.py` - pure-math checks of the FP integral
  (a uniform bulk profile must give FP=0; a depleted shell must give a
  nonzero FP) and the RDF-table -> per-atom-FP grouping logic.
- `test_calculate_rdf.py` - the bin-geometry helper (`make_bins`):
  bin/edge counts, Angstrom/nm scaling, shell-volume monotonicity.
- `test_g1_g2_selection.py` - the regex parser that turns the official
  selection algorithm's two printed result lines into structured G1/G2
  data, including noisy surrounding log text.
- `test_output_dir_creation.py` - **regression tests for a real bug**
  caught during manual end-to-end testing against a real ligand-in-water system:
  several scripts didn't create their output directory before writing,
  crashing with `OSError`/`FileNotFoundError` the first time anyone ran
  them into a fresh `outputs/` tree. Fixed in `select_g1_g2.py`,
  `build_ligand_cv.py`, `add_legend.py`, `prepare_ranking_csv.py`,
  `run_official_selection.py`, `calculate_rdf.py`, and
  `calculate_fingerprint.py`.

## What's intentionally NOT covered by automated tests

`prepare_ranking_csv.py`, `run_official_selection.py` (the vendored
official WaterFP selection algorithm), `monitor_convergence.py`, and
the waterfp `cli.py` all need a real `.tpr`/`.xtc` with actual bond
connectivity and a real water box - not something worth fabricating as
synthetic test fixtures, and real trajectory data doesn't belong in this
repository (see the top-level README's "code only" policy).

Instead, the full pipeline (stages 2 through 6) was validated by hand
against a real, already-existing ligand-only trajectory,
confirming an EXACT reproduction of the historical, previously-published
result: same convergence numbers (stop block, FP/RDF/Spearman metrics to
3 decimal places), same FP ranking table, same verbose selection trace,
same final G1/G2 atom pair (both anchors and both partners). That
validation is not committed here (it depends on data outside this repo)
but is exactly what caught the `os.makedirs` bug this test suite now
guards against, plus a documentation gap (Stage 3 must be run over the
convergence-defining block's own frame range, not the whole trajectory -
see `../src/ligand_waterfp/convergence/README.md`'s "Next stage" section).
