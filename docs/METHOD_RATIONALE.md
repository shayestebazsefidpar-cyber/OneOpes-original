# Method rationale

The FP of a ligand atom only means something once the water around it has equilibrated, and
only as precisely as the trajectory allows. The monitor therefore runs unbiased MD block by
block, and after each block checks where equilibration ended, how precise the FP of the
production part is, whether it still drifts, and whether the G1/G2 selection is settled.

## What is measured, per heavy atom and block

From WaterFP's `calc_rdf.sh` and `fp.py` (github.com/valeriorizzi/WaterFP), computed per block:

```
n(r) = water-O density in shells 0..rmax (2.001 nm, 0.001 nm bins)
bulk = mean n(r) over the last 500 bins          (no theoretical bulk density assumed)
g(r) = n(r) / bulk
FP   = ∫ -2π·bulk·(g ln g - g + 1)·r² dr          (trapezoid; the g→0 limit of the bracket is 1)
```

FP is the water excess-entropy density relative to bulk: 0 for bulk-like water, more negative
the more the atom perturbs it, whether water is denser or sparser. Atoms are then ranked by
FP (rank 1 = most negative); the G1/G2 selection uses this ranking.

Every frame, each atom's first-shell water count (waters within 0.35 nm) is also recorded.

## The check after each block

On all the blocks so far (1 ns each by default):

1. **Production start** (Chodera, *J. Chem. Theory Comput.* 2016; `pymbar.timeseries.detect_equilibration`).
   For each atom, the start of its first-shell count series that keeps the most uncorrelated
   samples. Production starts at the latest of these, at a block boundary, keeping at least
   the last two blocks.
2. **FP of production with its error** (block jackknife). FP is computed from the mean density of
   the n production blocks (`fp_uncorrected`), and again leaving out each block in turn
   (FP₋ⱼ, mean F̄). Then

   ```
   FP    = n·FP_all − (n−1)·F̄
   FP_SE = sqrt((n−1)/n · Σⱼ (FP₋ⱼ − F̄)²)
   ```

3. **Drift**. Each half of production gets its own jackknife; `drift_z` is the change of FP
   between the halves over the standard error of that change. Drift is reported when an atom's
   `drift_z` exceeds Student's t with n − 2 degrees of freedom at `drift_alpha` (5 %),
   Bonferroni-corrected over the atoms.
4. **Precision goal**, set by the command:
   - `monitor`: resample FP 1000 times from the jackknife (its covariance between atoms) and
     run the G1/G2 selection on each; the same pair must come out in at least
     `--selection-confidence` (95 %) of the draws;
   - `protein`: every FP standard error at most `--fp-rel-se-tol` (5 %) of its FP.

A check passes with at least `--min-blocks` (10) production blocks, no drift and the goal
met. Converged = `--n-passes` (2) passing checks in a row; the monitor then sends SIGTERM to
mdrun (the trajectory so far is kept) and selects G1/G2 on the production FP.

## Why not compare neighbouring blocks

The earlier criteria (FP change ≤ 10 %, g(r) NRMSD ≤ 0.15 and Spearman ≥ 0.9 between
consecutive 5 ns blocks, 3 times in a row) mostly measured noise. On the MAO-B real tests:

- FP has a finite-sampling bias of about −n_bins / (4·n_frames): from one 500-frame block,
  −1.0 of a ligand FP of about −3.3. A block's FP depended on how long the block was.
  The jackknife cancels a bias of this 1/n form exactly.
- The g(r) NRMSD of 0.001 nm bins scaled as 1/√frames: shot noise, not equilibration.
- The largest relative change over all atoms grew stricter with the number of atoms
  (22 ligand vs 171 protein atoms), and atoms with FP near 0 could decide it alone.
- A steady drift of a few percent per block passed every comparison of neighbouring blocks.
- The selection used one block, so its noise never shrank as the run grew.

## Limitations

- Production starts where the first-shell counts equilibrate; slower changes, such as the
  ligand's conformations, are only seen through drift. A ligand stuck in one rotamer can show
  converged hydration around an unrepresentative pose.
- The jackknife assumes blocks longer than the correlation time of the FP. First-shell waters
  exchange in picoseconds; slow pocket hydration in a protein can need longer blocks.
- The bulk density is estimated from the outer shells of the same frames, so box-density
  fluctuations (NPT) enter the error bars rather than being removed.
- The monitor checks after every block, which makes a lucky early pass more likely; the
  `--n-passes` and `--min-blocks` requirements guard against it.
- `--selection-confidence` assumes Gaussian FP errors. The thresholds are this workflow's
  choices, not from WaterFP (which defines FP, not convergence): tune them per system with
  `--min-blocks`, `--drift-alpha`, `--n-passes`, `--selection-confidence`, `--fp-rel-se-tol`
  and `--block-ns`.

## References

- Chodera, J. D. A simple method for automated equilibration detection in molecular
  simulations. *J. Chem. Theory Comput.* 12, 1799 (2016).
- Flyvbjerg, H. & Petersen, H. G. Error estimates on averages of correlated data.
  *J. Chem. Phys.* 91, 461 (1989).
- Grossfield, A. et al. Best practices for quantification of uncertainty and sampling quality
  in molecular simulations. *LiveCoMS* 1, 5067 (2018).
