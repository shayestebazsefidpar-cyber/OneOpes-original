# Official WaterFP selection (reference only)

Not a pipeline stage. Holds the upstream WaterFP code that
`oneopes.ligand.g1_g2` reproduces:

- **`vendored_selection.py`**: `load_mol()`, `path_length()`, `ranking()`,
  `select_next_atom()`, `select_bulk_atom()`, copied verbatim (whitespace
  aside) from `WaterFP_official_scripts/Scripts/fp_driven_atom_selection.ipynb`,
  cells 2 and 5. `tests/test_g1_g2_reference.py` runs them side by side
  with `select_g1_g2` on random ligands.
- **`WaterFP_official_scripts/`**: the vendored upstream files (`fp.py`,
  `calc_rdf.sh`, `fp_driven_atom_selection.ipynb`, `fp.yml`,
  `run_me_for_fp.sh`), with their provenance in its README.
