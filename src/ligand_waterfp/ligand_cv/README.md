# Stage 6 - Ligand-side CV construction

Builds a PLUMED input fragment from Stage 5's structured G1/G2 result:
`GROUP` definitions for G1 and G2, plus a `COORDINATION` CV for each
against a water-oxygen reference group.

- **Script**: `build_ligand_cv.py`
- **Input**: the `g1_g2.yaml` written by `ligand_waterfp.g1_g2_selection.select_g1_g2`
- **Output**: a PLUMED fragment (`.dat`) - a **starting point**, not a
  ready-to-run file

## Usage

```bash
ligand-waterfp-build-cv \
    --g1-g2 outputs/selection/g1_g2.yaml \
    --out outputs/ligand_cv/plumed_fragment.dat
```

## Important: review before production use

The generated fragment uses placeholder `COORDINATION` switching-function
parameters (`R_0`, `NN`, `MM`) and a placeholder water-reference group.
Before using this in a real `plumed.dat`:

1. Replace the water-oxygen group with your system's actual atom indices
   (pass `--water-group-def` to fill this in automatically), or reuse an
   existing group already defined elsewhere in your `plumed.dat`.
2. Review `R_0`/`NN`/`MM` against whatever convention the rest of your
   project's CVs already use, so the ligand-side CV behaves consistently
   with the protein-side / funnel CVs it will sit alongside.
3. Merge the fragment into your system's full `plumed.dat` by hand - this
   script deliberately does not attempt to merge into or overwrite an
   existing, system-specific `plumed.dat`.
