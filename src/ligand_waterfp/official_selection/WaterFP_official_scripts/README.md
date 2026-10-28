# Vendored upstream WaterFP scripts

The `Scripts/` folder in this directory is an **unmodified copy** of the
`Scripts/` folder from the official WaterFP repository:

- Upstream: https://github.com/valeriorizzi/WaterFP
- Vendored from commit: `6a5aef725a66b28e86d3582aee3e9e8bbbe4bc54`
- Files copied verbatim, byte-for-byte, no edits: `fp.py`, `calc_rdf.sh`,
  `fp_driven_atom_selection.ipynb`, `fp.yml`, `run_me_for_fp.sh`

These files are kept here purely for reference/provenance - to make clear
exactly which version of the official implementation
`run_official_selection.py` (one directory up) calls into. They are not
imported directly (a notebook cannot be imported as a Python module); the
`select_next_atom` / `select_bulk_atom` functions in
`fp_driven_atom_selection.ipynb` (cells 2 and 5) are reproduced verbatim,
with clear `BEGIN/END OFFICIAL CODE` markers and no logic changes, inside
`../run_official_selection.py`.

**License note**: no `LICENSE` file was present in the vendored commit.
Confirm licensing terms directly with the upstream repository/author
before redistributing this folder outside this project.

If the upstream repository is updated, re-vendor by replacing the five
files above from the new commit and updating the commit hash here and in
`../run_official_selection.py`'s docstring - do not hand-edit the vendored
files themselves.
