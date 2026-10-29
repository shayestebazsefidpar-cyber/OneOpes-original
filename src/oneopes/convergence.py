"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from itertools import accumulate, pairwise
from pathlib import Path

import numpy as np
import pandas as pd
from MDAnalysis.analysis.results import Results
from scipy.stats import spearmanr

from oneopes.waterfp import WaterFP


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
    """RMS difference of g(r) curves, relative to their highest peak; one value per atom (last axis = r)."""
    rmsd = np.sqrt(np.mean((g - prev_g) ** 2, axis=-1))
    peak = np.maximum(np.max(g, axis=-1), np.max(prev_g, axis=-1))
    return rmsd / np.maximum(peak, 1e-8)  # floor: g(r) that is zero everywhere


def check_stability(current: Results, previous: Results, tol: Tolerances) -> StabilityResult:
    """Compare two consecutive blocks' WaterFP results (same atoms, same order).

    Stable when, over all atoms, the largest FP change, the largest g(r) nRMSD
    and the Spearman correlation of the FP values are all within ``tol``. The
    relative FP change divides by |previous FP| floored at 1e-6, for atoms with FP near 0.
    """
    fp, prev_fp = current.fp, previous.fp
    fp_change = float(np.max(np.abs(fp - prev_fp) / np.maximum(np.abs(prev_fp), 1e-6)))
    rdf_nrmsd = float(np.max(nrmsd(current.g, previous.g)))
    rho = float(spearmanr(fp, prev_fp).statistic)

    fp_pass = fp_change <= tol.fp_rel_tol
    rdf_pass = rdf_nrmsd <= tol.rdf_nrmsd_tol
    spearman_pass = rho >= tol.spearman_tol
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


def summarize(stabilities: Sequence[StabilityResult], block_times: Sequence[tuple[float, float]],
              block_ns: float, tol: Tolerances) -> ConvergenceSummary:
    """Summary of a run of blocks; ``block_times[i]`` is the (start_ns, end_ns) of block i."""
    first = first_convergence(stabilities, tol.n_stable)
    block = None if first is None else first + 1
    return ConvergenceSummary(
        block_ns=block_ns,
        tolerances=tol,
        n_blocks=len(block_times),
        converged=block is not None,
        converged_block=block,
        converged_ns=None if block is None else block_times[block][1],
        unstable_after_convergence=0 if block is None else sum(not s.passed for s in stabilities[block:]),
        stable_in_a_row_at_end=(stable_in_a_row(stabilities) or [0])[-1],
    )


def write_summary(path: Path, summary: ConvergenceSummary, **facts) -> None:
    """Write ``summary`` and any other ``facts`` about the run as one indented JSON object."""
    path.write_text(json.dumps({**asdict(summary), **facts}, indent=2))


def fp_ranks(fp: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rank of each atom's FP (per column): 1 = most negative; ties keep the atoms' order."""
    return fp.rank(method="first").astype(int)


def stack_blocks(tables: Sequence[pd.DataFrame]) -> pd.DataFrame:
    """Each block's table one after the other, with the block index as first column."""
    stacked = pd.concat([table.assign(block=b) for b, table in enumerate(tables)], ignore_index=True)
    return stacked.reindex(columns=["block", *tables[0].columns])


def times_table(times_ns: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """One row per block: block, t_start_ns, t_end_ns, t_mid_ns; ``times_ns[i]`` is the (start, end) of block i."""
    start, end = np.array(times_ns, dtype=float).T
    return pd.DataFrame({"block": np.arange(len(start)), "t_start_ns": start, "t_end_ns": end,
                         "t_mid_ns": (start + end) / 2})


def fp_per_block(blocks: Sequence[WaterFP], times_ns: Sequence[tuple[float, float]],
                 labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (block, atom): block, t_start_ns, t_end_ns, t_mid_ns, atom, FP.

    ``times_ns[i]`` is the (start, end) of block i; ``labels`` name the atoms (default: atom names).
    """
    fp = stack_blocks([block.fp_table(labels) for block in blocks])
    return times_table(times_ns).merge(fp, on="block")


def rdf_per_block(blocks: Sequence[WaterFP], labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (block, atom, r): block, atom, r_nm, g_r. ``labels`` name the atoms (default: atom names)."""
    profiles = stack_blocks([block.profile_table(labels) for block in blocks])
    return profiles.drop(columns="n_r")


def transitions_table(stabilities: Sequence[StabilityResult], times_ns: Sequence[tuple[float, float]]) -> pd.DataFrame:
    """One row per block after the first: its times, its StabilityResult against the block before it,
    and the stable transitions in a row ending there.

    ``stabilities[i]`` compares block i + 1 with block i; ``times_ns[i]`` is the (start, end) of block i.
    """
    later_blocks = times_table(times_ns).iloc[1:len(stabilities) + 1].reset_index(drop=True)
    stability = pd.DataFrame(map(asdict, stabilities), columns=[field.name for field in fields(StabilityResult)])
    return later_blocks.join(stability).assign(stable_in_a_row=stable_in_a_row(stabilities))


def write_block_tables(outdir: Path, blocks: Sequence[WaterFP], times_ns: Sequence[tuple[float, float]],
                       tol: Tolerances, labels: Sequence[str] | None = None) -> None:
    """Write the block-wise CSVs to ``outdir``: FP and g(r) per block, FP and FP rank per atom and block,
    and the transitions between blocks (convergence_metrics.csv).

    ``times_ns[i]`` is the (start, end) of block i; ``labels`` name the atoms (default: atom names).
    """
    fp = fp_per_block(blocks, times_ns, labels)
    fp.to_csv(outdir / "fp_values_per_block.csv", index=False)
    rdf_per_block(blocks, labels).to_csv(outdir / "rdf_profiles_per_block.csv", index=False)
    fp_wide = fp.pivot(index="atom", columns="block", values="FP").loc[fp["atom"].unique()]
    fp_wide.to_csv(outdir / "fp_values_by_block_wide.csv")
    fp_ranks(fp_wide).to_csv(outdir / "fp_ranking_by_block_wide.csv")
    stabilities = compare_blocks([block.results for block in blocks], tol)
    transitions_table(stabilities, times_ns).to_csv(outdir / "convergence_metrics.csv", index=False)
