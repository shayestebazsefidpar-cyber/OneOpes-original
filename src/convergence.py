"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate, pairwise

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


def compare_blocks(blocks: Sequence[Results], tol: Tolerances) -> list[StabilityResult]:
    """Stability of each block against the one before it; transition i is block i -> i + 1."""
    return [check_stability(current, previous, tol) for previous, current in pairwise(blocks)]


def stable_in_a_row(stabilities: Sequence[StabilityResult]) -> list[int]:
    """For each transition, how many stable transitions in a row end there."""
    passes = (s.passed for s in stabilities)
    counts = accumulate(passes, lambda count, passed: count + 1 if passed else 0, initial=0)
    return list(counts)[1:]


def first_convergence(stabilities: Sequence[StabilityResult], n_stable: int) -> int | None:
    """Index of the transition completing the first run of n_stable stable ones, or None."""
    return next((i for i, count in enumerate(stable_in_a_row(stabilities)) if count >= n_stable), None)


def transitions_table(stabilities: Sequence[StabilityResult], block_times: Sequence[tuple[float, float]],
                      n_stable: int) -> pd.DataFrame:
    """One row per transition: the columns of convergence_metrics.csv.

    ``block_times[i]`` is the (start_ns, end_ns) of block i.
    """
    first = first_convergence(stabilities, n_stable)
    rows = []
    for i, (s, count) in enumerate(zip(stabilities, stable_in_a_row(stabilities))):
        start_ns, end_ns = block_times[i + 1]
        rows.append({
            "transition": f"{i}->{i + 1}",
            "previous_block": i,
            "block": i + 1,
            "block_start_ns": float(start_ns),
            "block_end_ns": float(end_ns),
            "FP_relative_change": s.fp_change,
            "FP_pass": s.fp_pass,
            "RDF_nRMSD": s.rdf_nrmsd,
            "RDF_pass": s.rdf_pass,
            "Spearman_rho": s.spearman_rho,
            "Spearman_pass": s.spearman_pass,
            "all_criteria_pass": s.passed,
            "stable_transition_count": count,
            "convergence_reached": first is not None and i >= first,
        })
    return pd.DataFrame(rows)
