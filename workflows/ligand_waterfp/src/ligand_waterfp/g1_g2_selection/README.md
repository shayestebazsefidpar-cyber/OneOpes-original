# G1/G2 result parsing (helper module, no CLI)

Turns the official selection algorithm's raw result sentences into a
small, structured G1/G2 result that downstream stages (and your own
analysis/reporting) can consume without re-parsing prose.

**This is no longer a separate pipeline stage.** `select_g1_g2.py` has no
`main()`/CLI of its own - `ligand_waterfp.official_selection.run_official_selection`'s
`main()` calls `parse_selection_lines()` and `write_g1_g2_yaml()` directly,
in-process, immediately after the vendored `select_next_atom()`/
`select_bulk_atom()` return. There is no intermediate text file and no
separate console script anymore (`ligand-waterfp-g1g2` was removed) - the
result sentences never leave that one process.

- **Module**: `select_g1_g2.py` - two pure functions, kept here (rather
  than inlined into `official_selection`) so they stay independently
  testable with synthetic text, with no dependency on a real
  `.tpr`/ranking CSV:
  - `parse_selection_lines(text)` - regexes the two result sentences into
    `{"G1": [...], "G2": [...]}`
  - `write_g1_g2_yaml(result, out_path, system_id=None)` - writes that to
    YAML, creating the output directory if needed

Output shape:

```yaml
G1:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
G2:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
```

This step does not re-run or reinterpret the selection algorithm itself
(that's entirely `official_selection`'s job) - it only reformats its
output.

## Usage

Not invoked directly - see
`../official_selection/README.md` (`ligand-waterfp-select --out
outputs/selection/g1_g2.yaml` now writes this YAML in one step).

`outputs/` is a per-run result and is git-ignored - this module's code
never hardcodes or commits an actual G1/G2 result for any specific
ligand.

## Next stage

`ligand_waterfp.ligand_cv.build_ligand_cv` consumes the YAML
`run_official_selection` writes to construct the ligand-side PLUMED CV
fragment.
