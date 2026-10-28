# Stage 3 - WaterFP and G1/G2 selection

Computes WaterFP for every ligand heavy atom over a frame window and picks
the two atom pairs the ligand-side CV is built on:

- **G1** (anti-bulk): the most negative FP atom, and a partner among the
  next most negative ones that is not bonded to it (the farthest one when
  FP values are within 0.2 after rounding).
- **G2** (bulk): the least negative FP atom, and the next least negative
  one at least three bonds away.

These are the rules of the upstream WaterFP notebook;
`tests/test_g1_g2_reference.py` checks them against its vendored code
(`../official_selection/vendored_selection.py`).

## Usage

The monitor already runs this on its converged block and writes the files
below to `<outdir>/selection`. Run the command by hand only for another
frame window (see `../convergence/README.md`'s "Next stage" for the
converged block's frames). The topology must carry bonds (a `.tpr`).

```bash
ligand-waterfp-select --tpr prod.tpr --xtc prod.xtc --outdir outputs/selection \
    --start-frame <start> --end-frame <end> [--ligand-resname MOL] [--fp-round-decimals 1]
```

Writes, for review:

- `rdf.csv` (`atom,r_nm,n_r`) - water-O number density per atom
- `fingerprints.csv` (`atom,fp`) - WaterFP per atom
- `g1_g2.yaml`:

```yaml
G1:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
G2:
  - {name: <atom>, serial: <n>}
  - {name: <atom>, serial: <n>}
```

`serial` is 1-based (MDAnalysis `Atom.index` + 1).

## Next stage

`ligand_waterfp.ligand_cv.build_ligand_cv` builds the PLUMED CV fragment
from `g1_g2.yaml`.
