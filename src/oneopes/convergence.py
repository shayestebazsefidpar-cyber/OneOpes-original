"""Block-wise convergence of WaterFP results: is the next block like the previous one?

A transition from one block to the next is stable when all three criteria are within the tolerances:

* ``largest_relative_fp_change``: no atom's FP changed by more than ``fp_rel_tol`` (relative);
* ``largest_rdf_nrmsd``: no atom's g(r) moved by more than ``rdf_nrmsd_tol`` (RMS change / peak height);
* ``fp_rank_correlation``: the atoms' FP ranking kept a Spearman correlation of at least ``spearman_tol``.

The run has converged at the transition completing the first ``n_stable`` stable transitions in a row.
Blocks are the stacked block measurements of :func:`oneopes.trajectory_blocks.stack_blocks`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import numpy as np
import xarray as xr
from scipy.stats import spearmanr

from oneopes.errors import InputError


@dataclass(frozen=True)
class Tolerances:
    """Thresholds for a stable transition, and how many in a row mean converged."""
    fp_rel_tol: float = 0.10
    """Largest relative FP change between consecutive blocks."""
    rdf_nrmsd_tol: float = 0.15
    """Largest normalised RMSD between consecutive g(r)."""
    spearman_tol: float = 0.90
    """Smallest Spearman correlation between consecutive FP rankings."""
    n_stable: int = 3
    """Stable transitions in a row that mean converged."""

    def __post_init__(self) -> None:
        if not (self.fp_rel_tol >= 0 and self.rdf_nrmsd_tol >= 0):
            raise InputError(f"FP and g(r) tolerances must be >= 0, got {self.fp_rel_tol} and {self.rdf_nrmsd_tol}.")
        if not -1 <= self.spearman_tol <= 1:
            raise InputError(f"The Spearman tolerance is a correlation, between -1 and 1; got {self.spearman_tol}.")
        if self.n_stable < 1:
            raise InputError(f"n_stable must be at least 1, got {self.n_stable}.")


DEFAULT_TOLERANCES = Tolerances()


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


def nrmsd(g: np.ndarray, prev_g: np.ndarray) -> np.ndarray:
    """RMS difference of g(r) curves relative to their highest peak; one value per atom (last axis = r).

    The peak is floored at 1e-8, for a g(r) that is zero everywhere.
    """
    rmsd = np.sqrt(np.mean((g - prev_g) ** 2, axis=-1))
    peak = np.maximum(np.max(g, axis=-1), np.max(prev_g, axis=-1))
    return rmsd / np.maximum(peak, 1e-8)


def largest_relative_fp_change(fp: np.ndarray, prev_fp: np.ndarray) -> float:
    """max over atoms of |FP - previous FP| / |previous FP|, the denominator floored at 1e-6 for FP near 0."""
    return float(np.max(np.abs(fp - prev_fp) / np.maximum(np.abs(prev_fp), 1e-6)))


def largest_rdf_nrmsd(g: np.ndarray, prev_g: np.ndarray) -> float:
    """max over atoms of the g(r) change, as :func:`nrmsd`."""
    return float(np.max(nrmsd(g, prev_g)))


def fp_rank_correlation(fp: np.ndarray, prev_fp: np.ndarray) -> float:
    """Spearman correlation of the atoms' FP values: 1 when the FP ranking of the atoms is unchanged."""
    return float(spearmanr(fp, prev_fp).statistic)


def check_stability(current: xr.Dataset, previous: xr.Dataset, tol: Tolerances) -> StabilityResult:
    """Compare two consecutive blocks' hydration (``fp`` and ``g`` of the same atoms) on the three criteria."""
    fp_change = largest_relative_fp_change(current.fp.values, previous.fp.values)
    rdf_nrmsd = largest_rdf_nrmsd(current.g.values, previous.g.values)
    rho = fp_rank_correlation(current.fp.values, previous.fp.values)

    fp_pass = fp_change <= tol.fp_rel_tol
    rdf_pass = rdf_nrmsd <= tol.rdf_nrmsd_tol
    spearman_pass = rho >= tol.spearman_tol
    return StabilityResult(
        fp_change=fp_change, rdf_nrmsd=rdf_nrmsd, spearman_rho=rho,
        fp_pass=fp_pass, rdf_pass=rdf_pass, spearman_pass=spearman_pass,
        passed=fp_pass and rdf_pass and spearman_pass,
    )


def compare_blocks(blocks: xr.Dataset, tol: Tolerances) -> list[StabilityResult]:
    """Stability of each block against the one before it; transition i is block i -> i + 1."""
    return [check_stability(blocks.isel(block=b), blocks.isel(block=b - 1), tol)
            for b in range(1, blocks.sizes["block"])]


def stable_in_a_row(stabilities: Sequence[StabilityResult]) -> list[int]:
    """For each transition, how many stable transitions in a row end there."""
    passes = (s.passed for s in stabilities)
    counts = accumulate(passes, lambda count, passed: count + 1 if passed else 0, initial=0)
    return list(counts)[1:]


def first_convergence(stabilities: Sequence[StabilityResult], n_stable: int) -> int | None:
    """Index of the transition completing the first run of n_stable stable ones, or None."""
    return next((i for i, count in enumerate(stable_in_a_row(stabilities)) if count >= n_stable), None)


@dataclass(frozen=True)
class ConvergenceSummary:
    """What comparing consecutive blocks concluded.

    ``converged_block`` is the block completing the first run of ``tolerances.n_stable``
    stable transitions and ``converged_ns`` its end time; both are None when there is none.
    """
    block_ns: float
    tolerances: Tolerances
    n_blocks: int
    converged: bool
    converged_block: int | None
    converged_ns: float | None
    unstable_after_convergence: int
    stable_in_a_row_at_end: int


def summarize(stabilities: Sequence[StabilityResult], blocks: xr.Dataset, block_ns: float,
              tol: Tolerances) -> ConvergenceSummary:
    """Summary of a run of blocks and of the transitions between them."""
    first = first_convergence(stabilities, tol.n_stable)
    block = None if first is None else first + 1
    return ConvergenceSummary(
        block_ns=block_ns,
        tolerances=tol,
        n_blocks=blocks.sizes["block"],
        converged=block is not None,
        converged_block=block,
        converged_ns=None if block is None else float(blocks.t_end_ns[block]),
        unstable_after_convergence=0 if block is None else sum(not s.passed for s in stabilities[block:]),
        stable_in_a_row_at_end=(stable_in_a_row(stabilities) or [0])[-1],
    )
