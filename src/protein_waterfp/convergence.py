"""
Convergence verdict over the blocks' WaterFP results.

Reuses the shared implementations unchanged:

* FP and g(r) of each block: ``waterfp.WaterFP`` (run by ``cli.run_analysis``).
* Stability of one block transition: ``convergence.check_stability`` with
  ``Tolerances`` (FP relative change, RDF nRMSD, Spearman; all three must pass).

Retrospective analysis of an existing trajectory: every block is analysed and
every transition is scored. Sustained convergence = ``n_stable`` consecutive
stable transitions; its first occurrence is reported, the analysis carries on
to the end of the trajectory.

Ranking convention everywhere: rank 1 = most negative FP (ascending FP,
ties broken by atom index).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
from MDAnalysis.analysis.results import Results
from convergence import Tolerances, check_stability

DEFAULT_TOLERANCES = Tolerances(fp_rel_tol=0.10, rdf_nrmsd_tol=0.15, spearman_tol=0.90, n_stable=3)


def fp_ranks(fp: dict[str, float], order: list[str]) -> pd.Series:
    """Rank 1 = most negative FP; ties broken by position in ``order``."""
    position = {a: i for i, a in enumerate(order)}
    ranked = sorted(fp, key=lambda a: (fp[a], position[a]))
    rank = {a: i + 1 for i, a in enumerate(ranked)}
    return pd.Series([rank[a] for a in order], index=order, dtype=float)


@dataclass
class ConvergenceTracker:
    """Retrospective convergence bookkeeping over consecutive block transitions.

    Every block of the supplied trajectory is added; nothing is stopped or
    truncated when convergence is reached. ``n_stable`` consecutive stable
    transitions only define *when* sustained convergence was first reached
    (the end time of the block completing that streak).
    """

    tol: Tolerances = field(default_factory=lambda: DEFAULT_TOLERANCES)
    streak: int = 0
    first_converged_block: int | None = None
    first_converged_ns: float | None = None
    rows: list = field(default_factory=list)
    _prev: Results | None = None

    def add_block(self, block: int, t_start_ns: float, t_end_ns: float, current: Results) -> None:
        """Score the transition from the previous block to ``current`` (a block's WaterFP results)."""
        if self._prev is not None:
            res = check_stability(current, self._prev, self.tol)
            self.streak = self.streak + 1 if res.block_stable else 0
            if self.streak >= self.tol.n_stable and self.first_converged_block is None:
                self.first_converged_block = block
                self.first_converged_ns = float(t_end_ns)
            self.rows.append({
                "transition": f"{block - 1}->{block}",
                "previous_block": block - 1,
                "block": block,
                "block_start_ns": float(t_start_ns),
                "block_end_ns": float(t_end_ns),
                "FP_relative_change": res.max_fp_relative_change,
                "FP_pass": res.value_stable,
                "RDF_nRMSD": res.max_rdf_nrmsd,
                "RDF_pass": res.rdf_stable,
                "Spearman_rho": res.spearman_rank_corr,
                "Spearman_pass": res.rank_stable,
                "all_criteria_pass": res.block_stable,
                "stable_transition_count": self.streak,
                "convergence_reached": self.first_converged_block is not None,
            })
        self._prev = current

    @property
    def status(self) -> str:
        return "CONVERGED" if self.first_converged_block is not None else "NOT_CONVERGED"

    def after_convergence(self) -> dict:
        """How the transitions after the first sustained convergence behaved."""
        if self.first_converged_block is None:
            return {"n_transitions_after_first_convergence": 0,
                    "n_unstable_transitions_after_first_convergence": 0,
                    "remained_stable_after_first_convergence": None}
        later = [r for r in self.rows if r["block"] > self.first_converged_block]
        unstable = sum(not r["all_criteria_pass"] for r in later)
        return {"n_transitions_after_first_convergence": len(later),
                "n_unstable_transitions_after_first_convergence": unstable,
                "remained_stable_after_first_convergence": unstable == 0}
