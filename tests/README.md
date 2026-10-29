# Tests

Automated unit/regression tests, using synthetic data only - no real
system, ligand, or scientific result appears here.

Run with:
```bash
pip install -e ".[dev]"
pytest
```

## What's covered here

- `test_waterfp.py` - the shared `oneopes.waterfp` module: radial grid geometry
  (bin/edge counts, shell-volume monotonicity), water counting on small
  synthetic universes (a water lands in the bin of its distance, waters
  beyond rmax are ignored, frame ranges add up, parallel = serial), and the
  FP integral (a uniform bulk profile gives FP=0; a depleted shell gives a
  negative FP).
- `test_system.py` - ligand, water oxygen and heavy-atom detection without
  a protein (hydrogens, virtual sites, explicit selections).
- `test_g1_g2_selection.py` - the G1/G2 result types and YAML file, and
  `ligand-waterfp-select` end to end on a synthetic ligand in water.
- `test_g1_g2_reference.py` - `select_g1_g2` picks the same atoms as the
  vendored WaterFP notebook code on hundreds of random ligands.
- `test_build_ligand_cv_consumer.py` - `build_ligand_cv.py`'s consumption
  of `g1_g2.yaml` through `SelectionResult.from_dict()`: valid input still
  produces the same serial strings, and a malformed file (missing G1/G2,
  or an atom entry missing `serial`) is rejected clearly.
- `test_output_dir_creation.py` - **regression tests for a real bug**
  caught during manual end-to-end testing against a real ligand-in-water system:
  several scripts didn't create their output directory before writing,
  crashing with `OSError`/`FileNotFoundError` the first time anyone ran
  them into a fresh `outputs/` tree. Fixed in `build_ligand_cv.py` and
  `add_legend.py`; `write_g1_g2_yaml` is covered in
  `test_g1_g2_selection.py`.

## What's intentionally NOT covered by automated tests

`monitor_convergence.py`'s main loop needs a real `.tpr`/`.xtc` with a
real water box - not something worth fabricating as synthetic test
fixtures, and real trajectory data doesn't belong in this repository (see
the top-level README's "code only" policy).

Before the selection was rewritten as `select_g1_g2`, the full pipeline
(then stages 2 through 6) was validated by hand
against a real, already-existing ligand-only trajectory,
confirming an EXACT reproduction of the historical, previously-published
result: same convergence numbers (stop block, FP/RDF/Spearman metrics to
3 decimal places), same FP ranking table, same verbose selection trace,
same final G1/G2 atom pair (both anchors and both partners). That
validation is not committed here (it depends on data outside this repo)
but is exactly what caught the `os.makedirs` bug this test suite now
guards against, plus a documentation gap (Stage 3 must be run over the
convergence-defining block's own frame range, not the whole trajectory -
see `../docs/ligand_monitor.md`'s "Next stage" section).
