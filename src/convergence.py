"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
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
    fp_change: float
    rdf_nrmsd: float
    spearman_rho: float
    fp_pass: bool
    rdf_pass: bool
    spearman_pass: bool
    passed: bool


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
    fp_change = float(np.max(np.abs(fp - prev_fp) / np.maximum(np.abs(prev_fp), 1e-6)))
    rdf_nrmsd = float(max(nrmsd(g, prev_g) for g, prev_g in zip(current.g, previous.g)))
    rho, _ = spearmanr(fp, prev_fp)  # ranks the FP values itself

    fp_pass = fp_change <= tol.fp_rel_tol
    rdf_pass = rdf_nrmsd <= tol.rdf_nrmsd_tol
    spearman_pass = bool(rho >= tol.spearman_tol)
    return StabilityResult(
        fp_change=fp_change, rdf_nrmsd=rdf_nrmsd, spearman_rho=rho,
        fp_pass=fp_pass, rdf_pass=rdf_pass, spearman_pass=spearman_pass,
        passed=fp_pass and rdf_pass and spearman_pass,
    )


@dataclass(frozen=True, eq=False)
class Block:
    """One block of the trajectory and its WaterFP results."""
    index: int
    start_ns: float
    end_ns: float
    results: Results


@dataclass(frozen=True, eq=False)
class Transition:
    """How a block compares with the one before it."""
    block: Block
    stability: StabilityResult
    stable_in_a_row: int  # consecutive stable transitions ending here


class ConvergenceTracker:
    """Compares each block with the previous one; converged after tol.n_stable stable transitions in a row."""

    def __init__(self, tol: Tolerances = DEFAULT_TOLERANCES) -> None:
        self.tol = tol
        self.transitions: list[Transition] = []
        self._previous: Block | None = None

    def add_block(self, block: Block) -> None:
        if self._previous is not None:
            stability = check_stability(block.results, self._previous.results, self.tol)
            in_a_row = self.stable_in_a_row + 1 if stability.passed else 0
            self.transitions.append(Transition(block, stability, in_a_row))
        self._previous = block

    @property
    def stable_in_a_row(self) -> int:
        return self.transitions[-1].stable_in_a_row if self.transitions else 0

    @property
    def first_convergence(self) -> Transition | None:
        """The transition that completed the first run of tol.n_stable stable ones."""
        return next((t for t in self.transitions if t.stable_in_a_row >= self.tol.n_stable), None)

    def after_convergence(self) -> dict:
        """How the transitions after the first convergence behaved."""
        first = self.first_convergence
        if first is None:
            return {"n_transitions_after_first_convergence": 0,
                    "n_unstable_transitions_after_first_convergence": 0,
                    "remained_stable_after_first_convergence": None}
        later = [t for t in self.transitions if t.block.index > first.block.index]
        unstable = sum(not t.stability.passed for t in later)
        return {"n_transitions_after_first_convergence": len(later),
                "n_unstable_transitions_after_first_convergence": unstable,
                "remained_stable_after_first_convergence": unstable == 0}


def transitions_table(tracker: ConvergenceTracker) -> pd.DataFrame:
    """One row per transition: the columns of convergence_metrics.csv."""
    first = tracker.first_convergence
    return pd.DataFrame([{
        "transition": f"{t.block.index - 1}->{t.block.index}",
        "previous_block": t.block.index - 1,
        "block": t.block.index,
        "block_start_ns": float(t.block.start_ns),
        "block_end_ns": float(t.block.end_ns),
        "FP_relative_change": t.stability.fp_change,
        "FP_pass": t.stability.fp_pass,
        "RDF_nRMSD": t.stability.rdf_nrmsd,
        "RDF_pass": t.stability.rdf_pass,
        "Spearman_rho": t.stability.spearman_rho,
        "Spearman_pass": t.stability.spearman_pass,
        "all_criteria_pass": t.stability.passed,
        "stable_transition_count": t.stable_in_a_row,
        "convergence_reached": first is not None and t.block.index >= first.block.index,
    } for t in tracker.transitions])
