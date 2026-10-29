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
- `test_g1_g2_selection.py` - the G1/G2 result types and YAML file, the
  selection rules on hand-made chains, and `oneopes select` end to end on a
  synthetic ligand in water (`ligand_systems.py`).
- `test_g1_g2_reference.py` - `select_g1_g2` picks the same atoms as the
  vendored WaterFP notebook code on hundreds of random ligands.
- `test_monitor_convergence.py` - `oneopes monitor` end to end on the same
  synthetic ligand: stopping at convergence and selecting G1/G2 on the
  converged block, or reading every block without converging.
- `test_build_ligand_cv_consumer.py` - `oneopes build-cv` reading
  `g1_g2.yaml`: valid input gives the expected serials, and a malformed file
  (missing G1/G2, or an atom entry missing `serial`) is rejected clearly.
- `test_output_dir_creation.py` - **regression tests for a real bug**
  caught during manual end-to-end testing against a real ligand-in-water system:
  several scripts didn't create their output directory before writing,
  crashing with `OSError`/`FileNotFoundError` the first time anyone ran
  them into a fresh `outputs/` tree. Fixed in `build_ligand_cv.py` and
  `add_legend.py`; `write_g1_g2_yaml` is covered in
  `test_g1_g2_selection.py`.

## What's intentionally NOT covered by automated tests

- Following a live mdrun: waiting for new frames and sending SIGTERM need a
  running GROMACS process (the monitor tests read a finished trajectory).
- `oneopes prepare-system`, which runs GROMACS.
- The PyMOL and Chimera render scripts, which run inside those programs.
- Results on real systems: real trajectories don't belong in this
  repository (see the top-level README's "code only" policy).
