# oneopes.protein - protein-side analysis (`oneopes protein`)

Protein-side WaterFP hydration analysis of one protein-ligand system, run
**retrospectively on an existing, completed unbiased MD trajectory**. The
trajectory length is read from the input (any length); the whole trajectory is
analysed and nothing is stopped when convergence is reached. It is
standalone and system-independent: every input is passed explicitly, and the
code contains no system-specific atoms, coordinates, walls or boundaries.

## Pipeline

```text
complex structure ──► protein heavy atoms within --candidate-cutoff-A (default 7 Å) of the ligand
                      (fixed candidate pool)
trajectory (length detected from the input)
           ──► consecutive --block-ns (default 5 ns) blocks over the whole trajectory
               per block and candidate: WaterFP / RDF, Contact (5 Å), Bridge (3.5 Å water)
           ──► metrics for every block transition ──► first sustained convergence
               (analysis continues to the end of the trajectory)
           ──► full-trajectory FP / Contact / Bridge + static SASA
           ──► final_per_atom_table.csv
```

* **Candidates**: protein (MDAnalysis `protein`) heavy atoms (not hydrogen,
  mass > 0; see `oneopes.system`) whose minimum distance to any ligand heavy
  atom in `--complex` is ≤ the cutoff. Computed once; never filtered afterwards.
* **WaterFP / RDF**: the shared `oneopes.waterfp.WaterFP` analysis, with the same
  grid options as the ligand commands: water-oxygen number density from 0 to
  `--rmax-nm` (2.001 nm) in `--binwidth-nm` (0.001 nm) bins, bulk density from
  the last `--norm-tail-bins` (500) bins.
* **Contact**: fraction of frames with the candidate within 5 Å of any ligand
  heavy atom. **Bridge**: fraction of frames with ≥1 water oxygen within
  3.5 Å of both the candidate and a ligand heavy atom.
* **SASA**: static, on the complex (protein + ligand heavy atoms, no
  water/ions), freesasa defaults.
* **Convergence**: the shared `oneopes.convergence.compare_blocks` with FP relative
  change ≤ 0.10, RDF nRMSD ≤ 0.15, Spearman ≥ 0.90 — all three in a
  transition (FP change and nRMSD are the worst case over all candidates).
  Sustained convergence = 3 consecutive stable transitions; the end time of
  the block completing the third one is `converged_ns`.
  This is only reported: every block up to the end of the trajectory is still
  analysed, so later loss of stability is visible in `convergence_metrics.csv`.
* **Times**: block *k* spans `t(first frame of k)` to that + block size;
  the trajectory runs from the time of its first frame to that of its last. Frames
  after the last complete block are not in any block but are included in the
  final values.

## Interpretation and limitations

* **What FP measures.** FP quantifies how much the local water-oxygen
  distribution around an atom deviates from bulk water. With the current
  WaterFP definition the integrand `-2π·norm·(g ln g − g + 1)·r²` is never
  positive, so FP ≤ 0 (FP = 0 for a bulk-like distribution).
* **More negative FP does not mean "more hydrated".** Buried or dry protein
  atoms can have strongly negative FP because the protein excludes water and
  strongly perturbs the distribution around them. `FP_Rank` 1 is the most
  negative FP, i.e. the strongest perturbation, not automatically the most
  hydrated atom. Choosing final hydration CVs requires further scientific
  interpretation (e.g. together with `Contact_occupancy_5A`,
  `Bridge_occupancy` and `SASA_A2`); this workflow does not select CVs.
* **Convergence criterion.** `fp_change` is a worst-case per-atom
  relative criterion, `max |FP_now − FP_prev| / max(|FP_prev|, 1e-6)` over
  all candidates. Atoms with FP close to zero (bulk-like, solvent-exposed) can
  produce very large relative changes and can dominate the verdict. This is
  the ligand-side criterion (the shared `oneopes.convergence.compare_blocks`),
  intentionally unchanged here.
* **Reference structure.** Candidate selection and `SASA_A2` use only the
  supplied `--complex` structure. SASA is static (one structure), not
  trajectory-averaged.
* **Distance definition.** `Min_distance_to_ligand_A` is the minimum distance
  from the protein atom to any ligand heavy atom, not to the ligand centre of
  mass.
* **Numbering.** `Atom_name` and `Residue_number` come from the topology
  universe (`--topology`); `Chain_ID` comes from the raw `--complex` file when
  it has chain IDs. Residue numbering can therefore differ between a `.tpr`
  and the PDB. `Atom_serial` is the atom's 1-based number in the topology
  (`Atom_index` + 1), as GROMACS and PLUMED count atoms; it is not the serial
  column of the PDB file, which can differ (e.g. after TER records).
* **Scope.** The workflow is retrospective post-processing of an existing
  trajectory. It does not run MD, OneOPES or PLUMED, and has no Zone or
  funnel logic.

## Usage

```bash
oneopes protein --complex complex.pdb --topology prod.tpr --trajectory prod.xtc \
    --output-dir outputs/run1
```

Options: `--candidate-cutoff-A`, `--first-frame`, `--last-frame` (exclusive),
`--ligand-selection`, `--water-selection`, and the same grid (`--rmax-nm`,
`--binwidth-nm`, `--norm-tail-bins`), `--block-ns` and tolerance flags as
`oneopes monitor` (`--fp-rel-tol`, `--rdf-nrmsd-tol`, `--spearman-tol`,
`--n-stable`).

The topology must carry masses and match the trajectory atom-for-atom. The
complex must have the same atom count and the same protein/ligand heavy-atom
names (water/hydrogen naming may differ between formats).

## Outputs (all in `--output-dir`)

| File | Content |
|---|---|
| `blocks.nc` | the whole block run as one xarray Dataset (NetCDF): per block and atom FP, g(r), density, bulk and the occupancies, each block's comparison with the one before, and the settings, time ranges and the run metadata as attributes; load it with `oneopes.block_run.load_run` |
| `candidate_atom_mapping.csv` | the candidate pool |
| `fp_values_per_block.csv`, `rdf_profiles_per_block.csv` | FP (with the block times) and g(r) per block and atom |
| `contact_occupancy_per_block.csv` | Contact / Bridge per block and atom |
| `fp_values_by_block_wide.csv`, `fp_ranking_by_block_wide.csv` | FP and rank (1 = most negative) per block |
| `convergence_metrics.csv` | one row per block after the first, compared with the block before: `block`, `fp_change`, `rdf_nrmsd`, `spearman_rho`, `fp_pass`, `rdf_pass`, `spearman_pass`, `passed`, `stable_in_a_row`, `t_start_ns`, `t_end_ns`, `n_frames`, `t_mid_ns` |
| `convergence_summary.json` | block size, tolerances, number of blocks, `converged_block`/`converged_ns`, unstable transitions after convergence, trajectory and analysed time range |
| `final_per_atom_table.csv` | one row per candidate, sorted by `FP_Rank` |
| `final_per_atom_table.md` | the same table as readable Markdown (floats rounded; the CSV keeps full precision) |
| `run_metadata.json` | inputs (file names, sha256), options, frames/blocks, versions |

Final table columns: `FP_Rank, Atom, Atom_index, Atom_serial, Atom_name,
Residue_name, Residue_number, Chain_ID, Min_distance_to_ligand_A, FP,
Contact_occupancy_5A, Bridge_occupancy, SASA_A2`. FP, Contact and Bridge are
computed over the full analysed frame range with the same code as the blocks;
SASA is static; `FP_Rank` 1 = most negative FP. Block-wise metrics are only in
`convergence_metrics.csv`; the analysed time range and the time of first
sustained convergence (`converged_ns`) are in `convergence_summary.json`.
