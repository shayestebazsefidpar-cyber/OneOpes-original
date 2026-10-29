"""Block-wise convergence of WaterFP results: is the next block like the previous one?

A transition from one block to the next is stable when all three criteria are within the tolerances:

* FP change: no atom's FP changed by more than ``fp_rel_tol`` (relative; the previous FP floored at 1e-6);
* g(r) change: no atom's g(r) moved by more than ``rdf_nrmsd_tol`` (:func:`nrmsd`);
* FP ranking: the atoms' FP values kept a Spearman correlation of at least ``spearman_tol``.

The run has converged at the block completing the first ``n_stable`` stable transitions in a row.
Blocks are the stacked block measurements of :func:`oneopes.trajectory_blocks.stack_blocks`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import accumulate

import xarray as xr
from scipy.stats import rankdata

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


def nrmsd(g: xr.DataArray, prev_g: xr.DataArray) -> xr.DataArray:
    """RMS change of each atom's g(r), relative to the higher peak of the two curves (floored at 1e-8)."""
    rmsd = xr.ufuncs.sqrt(((g - prev_g) ** 2).mean("r_nm"))
    peak = xr.ufuncs.maximum(g.max("r_nm"), prev_g.max("r_nm"))
    return rmsd / peak.clip(min=1e-8)


def atom_ranks(fp: xr.DataArray) -> xr.DataArray:
    """Rank of each atom's FP among the atoms (1 = most negative; ties share their average rank)."""
    return xr.apply_ufunc(rankdata, fp, input_core_dims=[["atom"]], output_core_dims=[["atom"]],
                          kwargs={"axis": -1})


def stable_in_a_row(passed: Sequence[bool]) -> list[int]:
    """For each transition, how many stable transitions in a row end there."""
    counts = accumulate(passed, lambda count, stable: count + 1 if stable else 0, initial=0)
    return list(counts)[1:]


TRANSITION_VARIABLES = ["fp_change", "rdf_nrmsd", "spearman_rho", "fp_pass", "rdf_pass", "spearman_pass", "passed",
                        "stable_in_a_row"]
"""The variables :func:`compare_blocks` adds."""


def compare_blocks(blocks: xr.Dataset, tol: Tolerances) -> xr.Dataset:
    """The blocks, each compared with the one before it.

    Adds the three criteria (``fp_change``, ``rdf_nrmsd``, ``spearman_rho``), whether each passed,
    whether all did (``passed``) and the stable transitions in a row ending at each block. Block 0
    has no block before it: its criteria are NaN, so it passes none and ends no stable run.
    """
    current = blocks[["fp", "g"]].isel(block=slice(1, None))
    previous = blocks[["fp", "g"]].shift(block=1).isel(block=slice(1, None))
    criteria = xr.Dataset({
        "fp_change": (abs(current.fp - previous.fp) / abs(previous.fp).clip(min=1e-6)).max("atom"),
        "rdf_nrmsd": nrmsd(current.g, previous.g).max("atom"),
        "spearman_rho": xr.corr(atom_ranks(current.fp), atom_ranks(previous.fp), dim="atom"),
    }).reset_coords(drop=True).reindex(block=blocks.block)
    fp_pass = criteria.fp_change <= tol.fp_rel_tol
    rdf_pass = criteria.rdf_nrmsd <= tol.rdf_nrmsd_tol
    spearman_pass = criteria.spearman_rho >= tol.spearman_tol
    passed = fp_pass & rdf_pass & spearman_pass
    return blocks.assign(criteria).assign(fp_pass=fp_pass, rdf_pass=rdf_pass, spearman_pass=spearman_pass,
                                          passed=passed, stable_in_a_row=("block", stable_in_a_row(passed.values)))


def converged_block(blocks: xr.Dataset, n_stable: int) -> int | None:
    """The block completing the first run of ``n_stable`` stable transitions, or None."""
    converged = blocks.block[blocks.stable_in_a_row >= n_stable]
    return int(converged[0]) if converged.size else None


def unstable_after(blocks: xr.Dataset, block: int) -> int:
    """How many transitions into the blocks after ``block`` were unstable."""
    return int((~blocks.passed).sel(block=slice(block + 1, None)).sum())


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


def summarize(blocks: xr.Dataset, block_ns: float, tol: Tolerances) -> ConvergenceSummary:
    """Summary of a run of blocks compared by :func:`compare_blocks`."""
    block = converged_block(blocks, tol.n_stable)
    return ConvergenceSummary(
        block_ns=block_ns,
        tolerances=tol,
        n_blocks=blocks.sizes["block"],
        converged=block is not None,
        converged_block=block,
        converged_ns=None if block is None else float(blocks.t_end_ns.sel(block=block)),
        unstable_after_convergence=0 if block is None else unstable_after(blocks, block),
        stable_in_a_row_at_end=int(blocks.stable_in_a_row[-1]),
    )
