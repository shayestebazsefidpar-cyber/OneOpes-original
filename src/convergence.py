"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

from dataclasses import dataclass

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
