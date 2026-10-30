# Protein-side analysis (`oneopes protein`)

For one protein-ligand system and a finished unbiased trajectory, measures for every protein
heavy atom near the ligand its WaterFP, contacts and SASA, block by block, and reports them over
the production blocks. Nothing is stopped at convergence; convergence is only reported.

```bash
oneopes protein --complex complex.pdb --topology prod.tpr --trajectory prod.xtc --output-dir outputs/protein
```

- **Candidates**: protein heavy atoms within `--candidate-cutoff-A` (7 Å) of a ligand heavy atom
  in `--complex`. Chosen once, never filtered.
- **WaterFP** per block (`--block-ns`, 1 ns) of the analysed range (`--first-frame`/`--last-frame`;
  at least two blocks), checked after each block as in `monitor` (`METHOD_RATIONALE.md`), with
  the same grid and criteria flags. The goal: every FP standard error at most `--fp-rel-se-tol`
  (5 %) of its FP.
- **Contact**: fraction of frames within 5 Å of a ligand heavy atom. **Bridge**: fraction of
  frames with a water oxygen within 3.5 Å of both the atom and the ligand.
- **SASA**: freesasa on the complex structure only (protein + ligand heavy atoms).

The topology must have masses and match the trajectory atom for atom. The complex must have the
same atoms, with the same protein and ligand heavy-atom names.

## Outputs

- `final_per_atom_table.csv` (and `.md`): one row per candidate, sorted by `FP_Rank`
  (1 = most negative FP), with FP (bias-corrected), `FP_SE`, `FP_uncorrected`, Contact and
  Bridge over the production blocks the last check found, and SASA. Frames after the last
  complete block are not used.
- `blocks.nc` and the per-block CSVs (FP, g(r), contacts, ranks; each atom with its residue,
  chain and distance to the ligand), `convergence_metrics.csv` (the check after each block),
  `convergence_summary.json` (`converged_ns` is the end of the block completing the passing
  checks; `production_start_ns` is where the last check found production to start).
- `fp_vs_time.png` and `g_per_block.png`, one panel for each of the 24 candidates with the most
  negative FP (any atoms: `notebooks/explore_run.py`).
- `candidate_atom_mapping.csv`, `run_metadata.json`.

## Reading the results

- FP ≤ 0, and 0 means bulk-like water. **More negative is not "more hydrated"**: a buried, dry
  atom can have a very negative FP because the protein excludes water. Read FP together with
  Contact, Bridge and SASA; this analysis does not pick CVs.
- The precision goal is relative, so an atom with FP near 0 can keep a run from converging.
- SASA and the candidate distances come from the one `--complex` structure, not the trajectory.
- `Atom_serial` is the 1-based topology number (as GROMACS/PLUMED count), not the PDB serial;
  `Residue_number` comes from the topology and can differ from the PDB's.
