# ligand_contact_map_generator

Standalone copy of the script that generates the PLUMED contact-map file
`cmap.dat` used by the OneOPES / metadynamics inputs in this project.

## Origin

* Source: `OneOpes-original/original/bin/ligand_contact_map_generator.py`
  (unchanged, byte-identical copy). Header: "(c) 2019 Antonija Kuzmanic".
  No separate license file accompanies the original script.
* It is the only code in the repository that writes `CONTACTMAP` blocks /
  `cmap.dat`. The `plumed.dat` files include the result
  (`INCLUDE FILE=../cmap.dat`) and the FES analysis scripts read the `cmap`
  CV (`run_fes_OneOPES_parallel.sh`, `funnel_FES_from_Reweighting.py
  --cv cmap`); they consume the file, they do not create it.

## What it does

From one reference structure (`.pdb` or `.gro`):

1. finds protein heavy atoms within `-cutoff` (nm) of the ligand selection
   (`mdtraj.compute_neighbors`); with `-include apolar` or a comma-separated
   residue list only those residue types are searched;
2. keeps one contact per neighbouring residue: the closest protein-heavy-atom /
   ligand-atom pair (`mdtraj.find_closest_contact`);
3. sorts the pairs by distance and writes the first `-max-contacts` of them as
   two PLUMED `CONTACTMAP` blocks with
   `SWITCH={RATIONAL R_0=0.55 D_0=0.00 NN=4 MM=10}`:
   `LABEL=cmap` (with `SUM`) and `LABEL=cmappa` (per-contact components);
4. writes a PyMOL script (`-vis`, default `show_dists.pml`) that draws the
   selected distances.

Atom numbers in `cmap.dat` are the structure file's atom serials (1-based),
i.e. PLUMED atom numbers when the structure has the simulation's atom order.

## Dependencies

* Python 3
* [mdtraj](https://www.mdtraj.org) (tested with mdtraj 1.11.1)

No other repository files are needed.

## Usage

```bash
python ligand_contact_map_generator.py -f whole.pdb \
    -lig-selection-string "resname LIG and not element H" \
    -cutoff 0.35 -o cmap.dat -vis show_dists.pml
```

| Option | Default | Meaning |
|---|---|---|
| `-f` (required) | – | reference structure, `.pdb` or `.gro` |
| `-lig-selection-string` (required) | – | mdtraj selection of the ligand heavy atoms |
| `-cutoff` | 0.3 | neighbour search cutoff (nm) |
| `-include` | `all` | `all`, `apolar` (GLY, ALA, VAL, LEU, ILE, PRO, PHE, MET, TRP) or a comma-separated resname list |
| `-max-contacts` | 5 | maximum number of contacts written |
| `-o` | `cmap.dat` | output PLUMED file |
| `-vis` | `show_dists.pml` | output PyMOL script |

Arguments are parsed at import time, so the file is meant to be run as a
script, not imported.

## Notes from validation

* Running the script on `original/HSP90/OneOPES/lig1/whole.pdb`
  (`"resname MOL and not element H"`, cutoff 0.35-0.45 nm) reproduces the 5
  atom pairs of the existing `original/HSP90/OneOPES/lig1/cmap.dat`; in the
  existing file contacts 4 and 5 (3.348 vs 3.355 A) appear in swapped order.
* For `original/BRD4/OneOPES/lig1/whole.pdb` the output has the identical
  format but contacts 2-4 differ from the existing BRD4 `cmap.dat`, so that
  file was produced from a different reference structure and/or settings or
  edited afterwards. The exact settings used for the archived `cmap.dat` files
  are not recorded.
* The script writes no trailing newline after the last `... CONTACTMAP`.
* As stated in the script, very large `.gro` files whose atom counter wraps
  are not supported.
