# Official WaterFP selection — HSP90 lig1 ligand

## 1. Scripts used (unmodified, from github.com/valeriorizzi/WaterFP, `Scripts/`)

- `Scripts/fp.py` — FP computation from `gmx rdf` output (number-density profile ->
  excess-entropy integral). Confirmed byte-for-byte identical formula to what our
  own `monitor_convergence.py`/`monitor_convergence_protein.py` already implement:
  `norm = mean(last 500 of 2001 g(r) bins)`, `g = y/norm`,
  `fp = trapz(-2*pi*norm*(g*ln(g)-g+1)*r^2, dx=1e-3 nm)` (with the `g=0` limiting
  case `-2*pi*norm*r^2`).
- `Scripts/calc_rdf.sh` — example `gmx rdf` driver (hardcoded to an 8-atom example
  ligand `C1-C6,O1,O2`; not run here, since we already have our own converged
  20 ns RDF data from `WaterFP_ligand/convergence/`).
- `Scripts/fp_driven_atom_selection.ipynb` — the actual **atom-selection**
  algorithm (`load_mol`, `path_length`, `ranking`, `select_next_atom`,
  `select_bulk_atom`), copied verbatim into `run_official_selection.py` between
  the `BEGIN/END OFFICIAL CODE` markers. **Not rewritten.**

Repo cloned into `./repo/` (shallow clone, HEAD at clone time).

## 2. System-specific adaptation (only inputs, not the algorithm)

- `ligand_fp_ranking_input.csv`: built from our own converged (t=15-20 ns,
  3 consecutive stable blocks) `fp_values_by_block_wide.csv` (final block = 3),
  with atom indices mapped to 1-based position in
  `MDAnalysis.Universe('HSP90lig1.tpr').select_atoms("resname MOL and not (name H*)")`
  — the same selection expression `load_mol()` uses internally, so indexing is
  consistent with the original notebook's convention.
- `HSP90lig1.tpr` — symlink to `../prod.tpr` (our own standalone ligand-in-water
  production run), named to satisfy `load_mol()`'s `mol.split('-')[0]+mol.split('-')[1]+'.tpr'`
  pattern for `mol = "HSP90-lig1"`.
- `fp_round`: not defined in the repo (their `ranking.csv` schema/precision isn't
  published); assumed `round(fp, 1)` since the algorithm's `0.2` grouping
  threshold only makes sense at ~1-decimal granularity. This is the one
  judgment call made; flagged here for scrutiny.

No files outside `WaterFP_ligand/WaterFP_official_scripts/` were touched.
`plumed.dat`, OneOPES, and the protein-side analysis were not touched.

## 3. Input FP values (final converged block, t=15-20 ns)

See `ligand_fp_ranking_input.csv`. Ranking (ascending FP):

```
N1 -20.16   C4 -19.91   C2 -19.85   C3 -19.78   C8 -19.67
C5 -19.63   C6 -19.62   C7 -19.61   C1 -19.59   C11 -19.52
C15 -19.46  C12 -19.40  C9 -19.39   C10 -19.32  C14 -19.26
C13 -19.23  O1 -18.99   O2 -18.83   O3 -18.70
```

## 4. Official algorithm output

**`select_next_atom` (anti-bulk fp pair)**: anchor = **N1** (highest-ranking,
fp_round -20.2). C1/C2/C3 rejected (bonded, path<=1). C4 provisionally wins,
then **C8 beats C4** (both within the 0.2 fp_round group, C8 has the longer
path: 3 vs 2). C5/C6 tried and lost (shorter path than C8). C7 ties C8 (same
path length, same fp) -> loop stops.
-> **N1 (1), C8 (10)**

**`select_bulk_atom` (bulk fp pair)**: anchor = **O3** (lowest-ranking,
fp_round -18.7). First atom scanned upward with path > 2 from O3 is **O2**
(path length 4) -> immediately accepted (function returns on first hit).
-> **O3 (19), O2 (14)**

| Atom | Role | FP | Path from its anchor | Reason selected |
|---|---|---|---|---|
| N1  | anti-bulk anchor | -20.16 | 0 | highest-ranking (most negative FP); <=2 atoms tied at that fp_round, no ambiguity |
| C8  | anti-bulk partner | -19.67 | 3 (from N1) | in N1's fp_round-0.2 group after C1-C3 rejected (bonded) and C4 (path 2) was beaten by longer-path C8 |
| O3  | bulk anchor | -18.70 | 0 | lowest-ranking (least negative / most bulk-like FP) |
| O2  | bulk partner | -18.83 | 4 (from O3) | first candidate scanning upward from the bottom with path > 2 from O3 |

## 5. Comparison with current PLUMED `G1`/`G2` (`rep_0/plumed.dat`, lines 36-37)

```
G1: GROUP ATOMS=1  #N1  MOL   210
G2: GROUP ATOMS=19  #S1  MOL   210
```

Verified atom identity by matching global atom numbering between
`original/HSP90/OneOPES/lig1/whole.pdb` (the bound complex) and
`WaterFP_ligand/prod.tpr` (the standalone ligand) — both enumerate the ligand
identically as atoms 1-32 (19 heavy + 13 H), atom 1 = N1, atom 19 = O3.

**G2's `#S1` comment is stale/wrong** — this ligand has no sulfur atom at all
(19 heavy atoms: C1-C15, N1, O1-O3). Atom index 19 is unambiguously O3 in
both topologies. This looks like the same kind of leftover copy-paste comment
found elsewhere in this project (e.g. `run_fes_OneOPES_parallel.sh`'s
`#Trypsin` comments) — a naming artifact, not a different atom.

**Result: G1 = N1 exactly matches the official algorithm's anti-bulk anchor.
G2 = O3 exactly matches the official algorithm's bulk anchor.** The existing
PLUMED setup already encodes the two *anchor* atoms the official method
selects. It does **not** include the two *partner* atoms (C8, O2) that
`select_next_atom`/`select_bulk_atom` also return — the current `plumed.dat`
tracks solvation of the two anchors only (via `NG1`, `NG2` COORDINATION CVs),
not the full anchor+partner quadruplet.

## 6. Critical verification: official vs. our local code

- **FP formula**: **identical**. Our `monitor_convergence.py`/
  `monitor_convergence_protein.py` already implement the exact same
  trapz-based excess-entropy integral as the official `fp.py`, down to the
  `norm`/`g=0` handling. This was verified by direct line-by-line comparison,
  not just docstring claim.
- **Atom selection/ranking (graph + path-distance criteria)**: our local
  scripts **never implemented this at all** — they only rank atoms by FP and
  run a Spearman-based block-convergence check. The graph/path-distance
  anchor+partner selection (`select_next_atom`, `select_bulk_atom`) is run
  here for the first time, using the unmodified official code.
- Conclusion: **not "different" in the sense of conflicting implementations**
  — the FP math was already correct and matching; the selection step was
  simply missing locally and has now been filled in with the real official
  code, not a reimplementation.

## 7. What was NOT done (per instructions)

`plumed.dat` untouched. `original/HSP90/OneOPES/` untouched. OneOPES not
rerun. No CVs changed. No new selection algorithm invented. WaterFP
thresholds (0.1 rel. change / 0.15 RDF nRMSD / 0.9 Spearman / 3 streak; here
the notebook's own 0.2 fp_round grouping and path<=1/<=2 rejection rules)
left as published. Protein side not analyzed in this step.
