# Stage 5 - G1/G2 result formalization

Turns Stage 4's raw text output into a small, structured G1/G2 result that
downstream stages (and your own analysis/reporting) can consume without
re-parsing prose.

- **Script**: `select_g1_g2.py`
- **Input**: the two result lines Stage 4 writes (via `--out`, or piped
  from stdout)
- **Output**: a YAML file, e.g.:

```yaml
G1:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
G2:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
```

This step does not re-run or reinterpret the selection algorithm itself
(that's entirely Stage 4's job) - it only reformats its output.

## Usage

```bash
ligand-waterfp-g1g2 \
    --selection-output outputs/selection/selection_output.txt \
    --out outputs/selection/g1_g2.yaml
```

`outputs/` here is a per-run result and is git-ignored - this stage's code
never hardcodes or commits an actual G1/G2 result for any specific
ligand.

## Next stage

`ligand_waterfp.ligand_cv.build_ligand_cv` consumes this YAML to construct the
ligand-side PLUMED CV fragment.
