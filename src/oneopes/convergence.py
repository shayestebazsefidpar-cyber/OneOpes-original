"""Statistical convergence of block-wise WaterFP: where production starts, FP with its error, and drift.

After each block, a check looks at all the blocks so far:

* production starts at the latest point, over the atoms, where their first-shell water counts have
  equilibrated (Chodera 2016), and keeps at least the last two blocks;
* FP of the production blocks together, corrected for its finite-sampling bias by a block jackknife,
  with its standard error;
* drift: each atom's FP change from the first half of production to the second, in standard errors;
* a precision goal, set by each command: a settled G1/G2 selection, or small enough FP errors.

A check passes with at least ``min_blocks`` production blocks, no drift and the goal met; the run
has converged at the block completing the first ``n_passes`` passing checks in a row.
docs/METHOD_RATIONALE.md explains the method.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from itertools import accumulate
from typing import Protocol

import numpy as np
import xarray as xr
from pymbar.timeseries import detect_equilibration
from scipy.stats import t as student_t

from oneopes.errors import InputError
from oneopes.waterfp import HydrationOptions, hydration_from_density


@dataclass(frozen=True)
class ConvergenceCriteria:
    """When a check passes, and how many passing checks in a row mean converged."""
    min_blocks: int = 10
    """Fewest production blocks a check trusts the jackknife with (at least 4: two for each half's drift)."""
    drift_alpha: float = 0.05
    """Chance of reporting drift, over all atoms together, in a run that has none."""
    n_passes: int = 2
    """Passing checks in a row that mean converged."""

    def __post_init__(self) -> None:
        if self.min_blocks < 4:
            raise InputError(f"min_blocks must be at least 4, two for each half's jackknife; got {self.min_blocks}.")
        if not 0 <= self.drift_alpha <= 1:
            raise InputError(f"drift_alpha is a probability, between 0 and 1; got {self.drift_alpha}.")
        if self.n_passes < 1:
            raise InputError(f"n_passes must be at least 1, got {self.n_passes}.")


DEFAULT_CRITERIA = ConvergenceCriteria()


class PrecisionGoal(Protocol):
    """How precise a command needs the FP before it can stop: a score of an estimate, and the scores that do."""

    @property
    def description(self) -> str: ...

    def score(self, estimate: xr.Dataset) -> float: ...

    def met(self, score: float) -> bool: ...


@dataclass(frozen=True)
class FpPrecision:
    """Every atom's FP standard error is at most ``rel_se_tol`` of its FP."""
    rel_se_tol: float = 0.05

    @property
    def description(self) -> str:
        return f"every FP standard error at most {self.rel_se_tol:.0%} of its FP"

    def score(self, estimate: xr.Dataset) -> float:
        """The largest standard error relative to its FP."""
        return float((estimate.fp_se / abs(estimate.fp)).max())

    def met(self, score: float) -> bool:
        return score <= self.rel_se_tol


def production_start(shell_count: xr.DataArray) -> int:
    """The block where production starts: the latest, over the atoms, at which equilibration ends (Chodera 2016).

    ``shell_count`` has dimensions block, frame and atom; equilibration can only end at the start of a block.
    """
    block_length = shell_count.sizes["frame"]
    series = shell_count.stack(time=("block", "frame")).transpose("atom", "time").values.astype(float)
    start_frame = max(detect_equilibration(atom_series, nskip=block_length)[0] for atom_series in series)
    return int(start_frame) // block_length


def production(blocks: xr.Dataset) -> xr.Dataset:
    """The production blocks of at least two ``blocks``: from where equilibration ends, but at least the last two."""
    start = min(production_start(blocks.shell_count), blocks.sizes["block"] - 2)
    return blocks.isel(block=slice(start, None))


def combined(blocks: xr.Dataset, options: HydrationOptions) -> xr.Dataset:
    """The hydration of equally long ``blocks`` together: g(r) and FP of their mean water density."""
    return hydration_from_density(blocks.density.mean("block"), options)


def jackknife(production: xr.Dataset, options: HydrationOptions) -> xr.Dataset:
    """The hydration of the production blocks together, with FP corrected for its bias and its standard error.

    Also holds ``fp_uncorrected``, the FP of the blocks together, and ``fp_left_out``, the FP without each block.
    """
    n = production.sizes["block"]
    together = combined(production, options)
    left_out_density = (production.density.sum("block") - production.density) / (n - 1)
    left_out = hydration_from_density(left_out_density, options).fp
    mean_left_out = left_out.mean("block")
    return together.assign(
        fp=n * together.fp - (n - 1) * mean_left_out,
        fp_se=np.sqrt((n - 1) / n * ((left_out - mean_left_out) ** 2).sum("block")),
        fp_uncorrected=together.fp,
        fp_left_out=left_out,
    )


def drift_z(production: xr.Dataset, options: HydrationOptions) -> xr.DataArray:
    """Each atom's FP change from the first half of production to the second, in standard errors of that change.

    Each half has its own jackknife, so it needs two blocks: with fewer than four, drift is NaN.
    """
    n = production.sizes["block"]
    if n < 4:
        return xr.full_like(production.fp.isel(block=0, drop=True), np.nan)
    first = jackknife(production.isel(block=slice(None, n // 2)), options)
    second = jackknife(production.isel(block=slice(n // 2, None)), options)
    change_se = np.sqrt(first.fp_se ** 2 + second.fp_se ** 2)
    return abs(second.fp - first.fp) / change_se.clip(min=1e-12)


def drift_limit(n_atoms: int, n_blocks: int, alpha: float) -> float:
    """The largest drift z of a run without drift: Student's t with the halves' n_blocks - 2 degrees of freedom,
    Bonferroni-corrected over the atoms."""
    return float(student_t.ppf(1 - alpha / (2 * n_atoms), df=n_blocks - 2))


ESTIMATE_VARIABLES = ["fp", "fp_se", "fp_uncorrected", "drift_z"]
"""The per-atom values of a production estimate."""


def production_estimate(production: xr.Dataset, options: HydrationOptions) -> xr.Dataset:
    """The jackknife of the production blocks, with each atom's drift."""
    return jackknife(production, options).assign(drift_z=drift_z(production, options))


def fp_samples(estimate: xr.Dataset, n_samples: int, rng: np.random.Generator) -> np.ndarray:
    """FP vectors (one row each) drawn around the estimate's FP, with the covariance of its jackknife."""
    left_out = estimate.fp_left_out.transpose("block", "atom")
    n = left_out.sizes["block"]
    deviations = (left_out - left_out.mean("block")).values
    return estimate.fp.values + np.sqrt((n - 1) / n) * rng.standard_normal((n_samples, n)) @ deviations


CHECK_VALUES = ["production_start_block", "production_start_ns", "n_production_blocks", "max_drift_z",
                "drift_limit", "goal_score"]
CHECK_RESULTS = ["enough_blocks", "no_drift", "goal_met", "passed"]
CHECK_VARIABLES = [*CHECK_VALUES, *CHECK_RESULTS]
"""The variables of a check: what it found, then what passed."""

UNCHECKED = xr.Dataset(dict.fromkeys(CHECK_VALUES, np.nan) | dict.fromkeys(CHECK_RESULTS, False))
"""The check of a single block: there is nothing to check yet."""


def check(blocks: xr.Dataset, criteria: ConvergenceCriteria, options: HydrationOptions,
          goal: PrecisionGoal) -> xr.Dataset:
    """The check after the last of ``blocks``: production, its drift against the limit, the goal, and what passed."""
    if blocks.sizes["block"] < 2:
        return UNCHECKED
    found = production(blocks)
    estimate = production_estimate(found, options)
    n_blocks = found.sizes["block"]
    max_drift, limit = float(estimate.drift_z.max()), drift_limit(found.sizes["atom"], n_blocks, criteria.drift_alpha)
    score = goal.score(estimate)
    passes = {"enough_blocks": n_blocks >= criteria.min_blocks, "no_drift": max_drift <= limit,
              "goal_met": goal.met(score)}
    return xr.Dataset({"production_start_block": int(found.block[0]),
                       "production_start_ns": float(found.t_start_ns[0]), "n_production_blocks": n_blocks,
                       "max_drift_z": max_drift, "drift_limit": limit, "goal_score": score,
                       **passes, "passed": all(passes.values())})


def check_every_block(blocks: xr.Dataset, criteria: ConvergenceCriteria, options: HydrationOptions,
                      goal: PrecisionGoal) -> list[xr.Dataset]:
    """The check after each block, on the blocks up to it."""
    return [check(blocks.isel(block=slice(n + 1)), criteria, options, goal) for n in range(blocks.sizes["block"])]


def passes_in_a_row(passed: Iterable[bool]) -> list[int]:
    """For each check, how many passing checks in a row end there."""
    counts = accumulate(passed, lambda count, passes: count + 1 if passes else 0, initial=0)
    return list(counts)[1:]


def with_checks(blocks: xr.Dataset, checks: Sequence[xr.Dataset]) -> xr.Dataset:
    """``blocks`` with the check after each block, and how many checks in a row passed up to it."""
    checked = xr.concat(checks, dim="block").assign_coords(block=blocks.block)
    return blocks.assign(checked).assign(passes_in_a_row=("block", passes_in_a_row(checked.passed.values)))


def checks_of(run: xr.Dataset) -> xr.Dataset:
    """The check after each block of a run, with its times."""
    return run[[*CHECK_VARIABLES, "passes_in_a_row"]]


def converged_block(run: xr.Dataset, n_passes: int) -> int | None:
    """The block completing the first ``n_passes`` passing checks in a row, or None."""
    converged = run.block[run.passes_in_a_row >= n_passes]
    return int(converged[0]) if converged.size else None


def failed_after(run: xr.Dataset, block: int) -> int:
    """How many checks after ``block`` failed."""
    return int((~run.passed).sel(block=slice(block + 1, None)).sum())


@dataclass(frozen=True)
class ConvergenceSummary:
    """What the checks concluded; ``converged_block`` and ``converged_ns`` are None if the run never converged.

    Production is as the last check found it.
    """
    block_ns: float
    criteria: ConvergenceCriteria
    n_blocks: int
    converged: bool
    converged_block: int | None
    converged_ns: float | None
    failed_after_convergence: int
    passes_in_a_row_at_end: int
    production_start_ns: float | None
    n_production_blocks: int | None


def summarize(run: xr.Dataset, block_ns: float, criteria: ConvergenceCriteria) -> ConvergenceSummary:
    """The convergence summary of a run with its checks."""
    block = converged_block(run, criteria.n_passes)
    last = run.isel(block=-1)
    checked = not np.isnan(last.production_start_block)
    return ConvergenceSummary(
        block_ns=block_ns,
        criteria=criteria,
        n_blocks=run.sizes["block"],
        converged=block is not None,
        converged_block=block,
        converged_ns=None if block is None else float(run.t_end_ns.sel(block=block)),
        failed_after_convergence=0 if block is None else failed_after(run, block),
        passes_in_a_row_at_end=int(last.passes_in_a_row),
        production_start_ns=float(last.production_start_ns) if checked else None,
        n_production_blocks=int(last.n_production_blocks) if checked else None,
    )
