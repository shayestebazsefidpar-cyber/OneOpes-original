# Method rationale

The FP of a ligand atom only means something once the water around it has equilibrated.
The monitor therefore runs unbiased MD block by block and stops once the hydration
around every heavy atom stops changing.

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

## When two consecutive blocks agree

All three must hold, because each alone can mislead: equal FP can hide a shifted RDF peak,
a stable RDF says nothing about FP, and the ranking can hold while all values drift.

| Criterion | Per atom, block n vs n-1 | Passes when |
|---|---|---|
| FP change | `abs(FP_n - FP_n-1) / max(abs(FP_n-1), 1e-6)` | max over atoms ≤ 0.10 |
| g(r) change | RMS difference / higher peak of the two curves | max over atoms ≤ 0.15 |
| Ranking | Spearman correlation of the atoms' FP | ≥ 0.90 |

Converged = 3 stable transitions in a row (a single match can be luck). Blocks are 5 ns.
With an mdrun PID, the monitor then sends SIGTERM; the trajectory so far is kept.

These thresholds, the block size and the run length are this workflow's own heuristics, not
from WaterFP (which defines FP, not convergence) or from an error analysis. Tune them per
system with `--fp-rel-tol`, `--rdf-nrmsd-tol`, `--spearman-tol`, `--n-stable`, `--block-ns`.

## Limitations

- Only the water structure converges here, not the ligand's conformations: a ligand stuck
  in one rotamer can show converged hydration around an unrepresentative pose.
- The bulk density is estimated per block from its own outer shells, so box-density
  fluctuations (NPT) move it too; only the run of consecutive passes guards against that.
- The relative FP change of an atom with FP near 0 can be large and decide the verdict alone.
