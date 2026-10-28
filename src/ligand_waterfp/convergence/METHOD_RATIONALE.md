# WaterFP-style hydration RDF/FP block-wise convergence monitor - method rationale

Reference method: https://github.com/valeriorizzi/WaterFP/tree/main/Scripts
(`calc_rdf.sh` + `fp.py` define how to compute RDF/FP; the block-wise
convergence/auto-stop logic below is **not** part of that repository and
was designed for this workflow specifically).

## 1. Why run this at all - the core hypothesis

A ligand's local hydration structure (how water oxygens are radially
arranged around each of its heavy atoms) is what the WaterFP method uses
to build a per-atom "fingerprint" (FP), used in later stages of this
workflow to help select which ligand atoms are good candidates for
hydration-monitoring collective variables (the G1/G2 pairs constructed in
the `g1_g2_selection` subpackage and the `ligand_cv` subpackage).

For that fingerprint to be meaningful, it must be computed from a
trajectory in which the ligand's local water shell has actually
equilibrated - i.e. water has had enough time to rearrange around every
heavy atom and settle into its steady-state radial distribution, given the
ligand's current conformation and position in the box.

The hypothesis behind this monitor is:

> Once the per-atom RDF shape, the per-atom scalar FP derived from it, and
> the FP-based ranking of all heavy atoms all stop changing block-to-block
> (within defined tolerances) over several consecutive blocks, the local
> hydration structure has converged, and further simulation time is not
> needed for hydration-fingerprint purposes.

This lets a production run stop as soon as hydration is converged, rather
than always running a fixed, possibly wasteful, duration.

This hypothesis says nothing about, and is NOT a substitute for, checking
convergence of other quantities (e.g. ligand conformational/torsional
sampling, RMSD, radius of gyration) - see "Limitations" below.

## 2. What is measured, per ligand heavy atom

**(a) RDF**: g(r) of that atom to all water oxygens, r in `[0, --rmax-nm]`
nm (default 2.001), bin width `--binwidth-nm` (default 0.001, i.e. 2001
bins) - matching WaterFP's `calc_rdf.sh` settings (`gmx rdf -bin 0.001
-norm number_density -rmax 2.001`) exactly, but recomputed per trajectory
block instead of once over a whole fixed trajectory (see
`ligand_waterfp.waterfp.calculate_rdf`).

Per block: `n(r)` = mean neighbour count per radial shell per frame,
divided by the shell volume (waters/nm^3) - a raw, un-normalised
number-density profile.

**(b) FP (fingerprint)**: reimplemented exactly from WaterFP's `fp.py`
(see `ligand_waterfp.waterfp.calculate_fingerprint`):

```
norm  = mean of the last --norm-tail-bins bins (default 500 of 2001)
        (empirical bulk-water-density plateau; no external/theoretical
        bulk density value is assumed)
g(r)  = n(r) / norm                      (dimensionless RDF)
FP    = integral_0^{rmax}  -2*pi*norm*(g*ln(g) - g + 1)*r^2 dr
        (trapezoidal rule; where g=0 the integrand's well-defined limit,
        1, is substituted for (g*ln(g)-g+1))
```

This integrand is the standard water excess/relative-entropy density
relative to bulk (same functional form used in inhomogeneous solvation
theory / GIST-style analyses): it grows wherever a heavy atom's presence
makes local water systematically denser or sparser than bulk, either
direction contributing a nonzero (negative, by this sign convention)
contribution to FP. It is NOT a simple coordination number - two atoms can
have similar coordination but different FP if the shape of their water
shell differs from a bulk-like distribution.

**(c) Ranking**: within each block, all ligand heavy atoms are ranked by
FP (rank 1 = most negative / most bulk-perturbing, in this sign
convention). This ranking, not just the raw FP scalar, is what the
downstream atom-selection step (the `official_selection` subpackage) actually
uses.

## 3. Why three separate stability checks, not one

A single criterion can be misleading on its own:

- FP scalar alone could coincidentally match between two blocks while the
  underlying RDF shape is still drifting (e.g. a shifted peak that
  integrates to the same value).
- RDF shape alone says nothing about whether the derived FP values (what
  matters for ranking/selection) have actually stabilised.
- Ranking alone can look "stable" even while absolute values are still
  drifting substantially, if all atoms drift together.

Requiring RDF shape, FP value, AND ranking to be simultaneously stable is
a stronger, more defensible convergence claim than any one of them alone -
this is the compound criterion actually implemented in
`monitor_convergence.py`.

**RDF-profile stability** (per atom, block n vs n-1):
```
RMSD_i  = sqrt( mean_k ( g_i^n(r_k) - g_i^(n-1)(r_k) )^2 )
nRMSD_i = RMSD_i / max( max_k g_i^n(r_k), max_k g_i^(n-1)(r_k), 1e-8 )
criterion: max_i nRMSD_i <= --rdf-nrmsd-tol (default 0.15, i.e. 15%)
```
The RMSD between the two curves is normalised by the taller of the two
curves' own peak height (their first-hydration-shell peak), so nRMSD reads
as "the curves differ by at most X% of the peak height." This exact
normalisation choice is this workflow's own design decision - it is not
part of the WaterFP reference repository, which does not define any
block-stability metric at all.

**FP stability** (per atom, block n vs n-1):
```
rel_change_i = |FP_i^n - FP_i^(n-1)| / max(|FP_i^(n-1)|, 1e-6)
criterion: max_i rel_change_i <= --fp-rel-tol (default 0.10, i.e. 10%)
```

**Ranking stability** (block n vs n-1):
```
rho = Spearman rank correlation between the two blocks' FP-based atom
      rankings (scipy.stats.spearmanr)
criterion: rho >= --spearman-tol (default 0.90)
```

A block-transition is "stable" only if all three criteria hold at once.

## 4. The stopping rule

A single stable transition could be a coincidence (e.g. two noisy blocks
happening to agree). The workflow therefore requires `--n-stable`
(default 3) CONSECUTIVE stable block-transitions in a row before declaring
convergence - a lucky one-off match is very unlikely to repeat that many
times running.

Block size: `--block-ns` (default 5.0 ns) of newly-accumulated trajectory
per check. This is a practical choice (fast enough to check convergence
often; long enough for a typical few-thousand-water box to give a
reasonably converged per-block RDF/FP estimate on its own) - not a value
taken from the WaterFP reference, and worth revisiting for a much
smaller/larger solvent box.

On convergence, if a running `mdrun` PID was passed, the monitor sends
`SIGTERM` to stop production early; whatever trajectory exists up to that
point is kept as the final result.

**All numeric thresholds above (RDF nRMSD, FP change, Spearman rho,
consecutive-transitions count, block size) are reasonable,
transparently-documented defaults, adjustable via CLI flags. They are NOT
specified anywhere in the WaterFP reference repository** (which only
defines how to compute RDF/FP, not how to judge block-to-block
convergence) **and are not derived from any statistical error analysis**
- treat them as a starting heuristic, tune per system if a stricter or
looser convergence definition is wanted.

## 5. Limitations / what this does not claim

- This checks convergence of WATER STRUCTURE around a fixed ligand only.
  It does not check whether the ligand itself has adequately sampled its
  own conformational/torsional space - a ligand frozen in one rotamer for
  the whole (short) run could show perfectly "converged" hydration around
  that one conformation without that conformation being representative.
- The FP integral's bulk reference ("norm") is estimated per atom, per
  block, from that block's own outer-shell density - a noisy box density
  fluctuation (e.g. under NPT) could shift norm block-to-block independent
  of any real change in local structure; this is only implicitly guarded
  against by requiring several consecutive passes.
- Convergence of the FP/ranking does not itself constitute or perform G1/G2
  atom selection, CV construction, or any other downstream analysis -
  that is the `official_selection` subpackage onward, explicitly separate
  stages.

## 6. What the code actually does, block by block

```
for each new block of trajectory (--block-ns):
    for each ligand heavy atom:
        trajectory -> raw water-O density profile n(r)     [calculate_rdf.py]
                   -> g(r) = n(r) / (block's own bulk-tail estimate)
                   -> FP   = entropy-integral of g(r)       [calculate_fingerprint.py]
    rank all atoms by FP

    if a previous block exists:
        compare current vs. previous block:
            Delta_FP  for every atom      (section 3, FP stability)
            RDF nRMSD for every atom      (section 3, RDF stability)
            Spearman rho of the rankings  (section 3, ranking stability)
        block_transition_stable = (all three criteria pass)
        stable_streak = stable_streak + 1 if stable else reset to 0

    if stable_streak >= --n-stable:
        declare convergence, record stop block/time,
        send SIGTERM to the running mdrun process (if given), stop monitoring
```
