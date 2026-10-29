# Tests

```bash
pip install -e ".[dev]" && pytest
```

Synthetic data only; each file's docstring says what it covers. `tests/reference/` holds the
vendored WaterFP selection code the G1/G2 rules are checked against.

Not covered: GROMACS itself (`prepare-system` runs with `echo` in place of gmx), running the
PyMOL render script, and the BioSimSpace tests when BioSimSpace is not installed.
