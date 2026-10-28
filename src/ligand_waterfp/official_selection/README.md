# Stage 4 - Official WaterFP atom selection

Runs the **official, upstream** WaterFP atom-selection algorithm against
your converged FP ranking (from Stage 3). This selection logic is
deliberately not reimplemented - it is reproduced verbatim from the
upstream repository, with clear provenance markers, and must always be
traced back there. See `WaterFP_official_scripts/README.md` for the exact
vendored commit.

- **`run_official_selection.py`**: the driver. Everything between its
  `BEGIN/END OFFICIAL CODE` markers is vendored from
  `WaterFP_official_scripts/Scripts/fp_driven_atom_selection.ipynb` (cells
  2 and 5) - `load_mol()`, `path_length()`, `ranking()`,
  `select_next_atom()`, `select_bulk_atom()`, all five, with their
  original signatures, logic, and variable names unchanged (`ranking()` is
  a real dependency `select_next_atom`/`select_bulk_atom` call internally,
  not optional - see the script's own docstring for the diff confirming
  only whitespace/PEP8 formatting differs from the raw notebook cells).
  Only `main()`, below the marked block, is new - and `main()` now also
  parses the two result sentences and writes the structured G1/G2 YAML
  directly (via `ligand_waterfp.g1_g2_selection.select_g1_g2`), all in
  this same process, if `--out` is given. There is no intermediate text
  file and no separate CLI stage for that anymore.
- **`prepare_ranking_csv.py`**: builds this stage's required input CSV
  from Stage 3's `fingerprints.csv`. The official notebook never publishes
  how its own `ranking.csv` was built from raw FP values (cell 1 just
  reads a pre-made file) - only how it's consumed. This script is
  necessary glue code, not part of the official algorithm.
- **`WaterFP_official_scripts/`**: the vendored upstream files
  (`fp.py`, `calc_rdf.sh`, `fp_driven_atom_selection.ipynb`, `fp.yml`,
  `run_me_for_fp.sh`), kept for reference/provenance.

## Input format

`run_official_selection.py --ranking-csv` expects one row per ligand heavy
atom:

```
name,atom,fp,fp_round
<system_id>,<1-based atom serial in the .tpr>,<FP value>,<rounded FP>
```

`prepare_ranking_csv.py` builds exactly this from Stage 3's
`fingerprints.csv` (`atom,fp`): it derives each atom's 1-based serial from
`Universe(tpr).select_atoms("resname <ligand_resname> and not name H*")` -
the same selection expression `load_mol()` uses internally, so the serials
line up with what `select_next_atom`/`select_bulk_atom` index into - and
fills `fp_round = round(fp, --fp-round-decimals)` (see that script's
docstring for why this default is a reasonable placeholder, not a
confirmed reproduction of the upstream authors' own convention).

## Usage

`--system-id` must contain a hyphen (`<project>-<ligand>`): the vendored
`load_mol()` derives the `.tpr` filename it opens as
`mol.split('-')[0]+mol.split('-')[1]+'.tpr'` - this is the official code's
own convention, not something this workflow introduced, so `myproject.tpr`
(matching Stage 1/2/3's ligand topology, symlinked/copied/renamed to that
exact name in your working directory) must exist for `--system-id
myproject-ligandA`.

```bash
ligand-waterfp-prepare-ranking-csv \
    --tpr outputs/lig_system/prod.tpr \
    --fp-csv outputs/fingerprints.csv \
    --system-id myproject-ligandA \
    --out outputs/selection/ranking_input.csv

# load_mol() needs '<system-id, hyphen-parts joined>.tpr' in the CWD:
ln -s outputs/lig_system/prod.tpr myprojectligandA.tpr

ligand-waterfp-select \
    --ranking-csv outputs/selection/ranking_input.csv \
    --system-id myproject-ligandA \
    --out outputs/selection/g1_g2.yaml
```

## Output

Prints the anti-bulk pair (this stage's raw "G1") and the bulk pair
("G2") result sentences to stdout for inspection:

```
anti-bulk fp selection: <atom> (<serial>), <atom> (<serial>)
bulk fp selection: <atom> (<serial>), <atom> (<serial>)
```

and, if `--out` is given, parses them in-process and writes the
structured G1/G2 result straight to that YAML path (via
`ligand_waterfp.g1_g2_selection.select_g1_g2.write_g1_g2_yaml`) - no
intermediate text file, no separate parsing stage.
