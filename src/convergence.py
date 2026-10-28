"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

from dataclasses import dataclass, field

import numpy as np
from MDAnalysis.analysis.results import Results
from scipy.stats import spearmanr


@dataclass(frozen=True)
class Tolerances:
    """Thresholds for a stable transition, and how many in a row mean converged."""
    fp_rel_tol: float
    rdf_nrmsd_tol: float
    spearman_tol: float
    n_stable: int


DEFAULT_TOLERANCES = Tolerances(fp_rel_tol=0.10, rdf_nrmsd_tol=0.15, spearman_tol=0.90, n_stable=3)


@dataclass(frozen=True)
class StabilityResult:
    """The three metrics of one transition, whether each passed, and whether all did."""
    max_fp_relative_change: float
    max_rdf_nrmsd: float
    spearman_rank_corr: float
    value_stable: bool
    rdf_stable: bool
    rank_stable: bool
    block_stable: bool


def nrmsd(g_a, g_b):
    """RMS difference of two g(r) curves, relative to their highest peak."""
    rmsd = np.sqrt(np.mean((g_a - g_b) ** 2))
    norm = max(np.max(g_a), np.max(g_b), 1e-8)
    return rmsd / norm


def check_stability(current: Results, previous: Results, tol: Tolerances) -> StabilityResult:
    """Compare two consecutive blocks' WaterFP results (same atoms, same order).

    Stable when, over all atoms, the largest FP change, the largest g(r) nRMSD
    and the FP rank correlation are all within ``tol``.
    """
    fp, prev_fp = current.fp, previous.fp
    max_rel_change = float(np.max(np.abs(fp - prev_fp) / np.maximum(np.abs(prev_fp), 1e-6)))
    max_nrmsd = float(max(nrmsd(g, prev_g) for g, prev_g in zip(current.g, previous.g)))
    rho, _ = spearmanr(fp, prev_fp)  # ranks the FP values itself

    value_ok = max_rel_change <= tol.fp_rel_tol
    rank_ok = bool(rho >= tol.spearman_tol)
    rdf_ok = max_nrmsd <= tol.rdf_nrmsd_tol
    block_stable = value_ok and rank_ok and rdf_ok

    return StabilityResult(
        max_fp_relative_change=max_rel_change,
        max_rdf_nrmsd=max_nrmsd,
        spearman_rank_corr=rho,
        value_stable=value_ok,
        rdf_stable=rdf_ok,
        rank_stable=rank_ok,
        block_stable=block_stable,
    )


@dataclass
class ConvergenceTracker:
    """Scores each new block against the previous one and records when convergence is first reached.

    Converged = ``tol.n_stable`` stable transitions in a row; later blocks are still scored.
    """

    tol: Tolerances = DEFAULT_TOLERANCES
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
        """How the transitions after the first convergence behaved."""
        if self.first_converged_block is None:
            return {"n_transitions_after_first_convergence": 0,
                    "n_unstable_transitions_after_first_convergence": 0,
                    "remained_stable_after_first_convergence": None}
        later = [r for r in self.rows if r["block"] > self.first_converged_block]
        unstable = sum(not r["all_criteria_pass"] for r in later)
        return {"n_transitions_after_first_convergence": len(later),
                "n_unstable_transitions_after_first_convergence": unstable,
                "remained_stable_after_first_convergence": unstable == 0}
