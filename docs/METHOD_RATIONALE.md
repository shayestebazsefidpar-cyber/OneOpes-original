# Method rationale

The FP of a ligand atom only means something once the water around it has equilibrated, and
only as precisely as the trajectory allows. The monitor therefore runs unbiased MD block by
block, and after each block checks where equilibration ended, how precise the FP of the
production part is, whether it still drifts, and whether the G1 and G2 anchors are settled.

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
   The start of the atoms' total first-shell count series that keeps the most uncorrelated
   samples, at a block boundary, keeping at least the last two blocks. One total, not the
   latest start over the atoms: on the real ligand, 20 of 22 atoms started at 0 ns and one
   noisy atom at 4 ns, and the latest start jumped between 1 and 6 ns from check to check.
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
   - `monitor`: draw FP 1000 times from the jackknife (its covariance between atoms); the G1
     and G2 anchors, the most and least negative FP atoms, must be the same in at least
     `--anchor-confidence` (95 %) of the draws. The partners are left out: on the real ligand
     the G1 partner was C16 (FP −2.635) or C17 (−2.620), less than one standard error (0.02)
     apart, and telling them apart would take about 25 times more sampling. The selection
     of every draw is written to `g1_g2_draws.csv` (there: C16 67 %, C17 27 %, C8 5 %);
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
- The FP draws assume Gaussian errors. The thresholds are this workflow's choices, not from
  WaterFP (which defines FP, not convergence): tune them per system with `--min-blocks`,
  `--drift-alpha`, `--n-passes`, `--anchor-confidence`, `--fp-rel-se-tol` and `--block-ns`.

## References

- Chodera, J. D. A simple method for automated equilibration detection in molecular
  simulations. *J. Chem. Theory Comput.* 12, 1799 (2016).
- Flyvbjerg, H. & Petersen, H. G. Error estimates on averages of correlated data.
  *J. Chem. Phys.* 91, 461 (1989).
- Grossfield, A. et al. Best practices for quantification of uncertainty and sampling quality
  in molecular simulations. *LiveCoMS* 1, 5067 (2018).
