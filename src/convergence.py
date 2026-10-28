"""Block-wise convergence of WaterFP results: is the next block like the previous one?"""

from collections.abc import Sequence
from dataclasses import asdict, dataclass
from itertools import accumulate, pairwise

import numpy as np
import pandas as pd
from MDAnalysis.analysis.results import Results
from scipy.stats import spearmanr

from waterfp import WaterFP


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


def nrmsd(g: np.ndarray, prev_g: np.ndarray) -> np.ndarray:
    """RMS difference of g(r) curves, relative to their highest peak; one value per atom (last axis = r)."""
    rmsd = np.sqrt(np.mean((g - prev_g) ** 2, axis=-1))
    peak = np.maximum(np.max(g, axis=-1), np.max(prev_g, axis=-1))
    return rmsd / np.maximum(peak, 1e-8)  # floor: g(r) that is zero everywhere


def check_stability(current: Results, previous: Results, tol: Tolerances) -> StabilityResult:
    """Compare two consecutive blocks' WaterFP results (same atoms, same order).

    Stable when, over all atoms, the largest FP change, the largest g(r) nRMSD
    and the FP rank correlation are all within ``tol``.
    """
    fp, prev_fp = current.fp, previous.fp
    fp_change = float(np.max(np.abs(fp - prev_fp) / np.maximum(np.abs(prev_fp), 1e-6)))  # floor: FP of ~0
    rdf_nrmsd = float(np.max(nrmsd(current.g, previous.g)))
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


def fp_ranks(fp: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rank of each atom's FP (per column): 1 = most negative; ties keep the atoms' order."""
    return fp.rank(method="first").astype(int)


def fp_per_block(blocks: Sequence[WaterFP], times_ns: Sequence[tuple[float, float]],
                 labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (block, atom): block, t_start_ns, t_end_ns, t_mid_ns, atom, FP.

    ``times_ns[i]`` is the (start, end) of block i; ``labels`` name the atoms (default: atom names).
    """
    tables = [block.fp_table(labels).assign(block=b, t_start_ns=start, t_end_ns=end, t_mid_ns=0.5 * (start + end))
              for b, (block, (start, end)) in enumerate(zip(blocks, times_ns))]
    return pd.concat(tables, ignore_index=True)[["block", "t_start_ns", "t_end_ns", "t_mid_ns", "atom", "FP"]]


def rdf_per_block(blocks: Sequence[WaterFP], labels: Sequence[str] | None = None) -> pd.DataFrame:
    """One row per (block, atom, r): block, atom, r_nm, g_r. ``labels`` name the atoms (default: atom names)."""
    tables = [block.profile_table(labels).assign(block=b) for b, block in enumerate(blocks)]
    return pd.concat(tables, ignore_index=True)[["block", "atom", "r_nm", "g_r"]]


# Column name of each StabilityResult field in convergence_metrics.csv, in column order.
CSV_COLUMNS = {
    "fp_change": "FP_relative_change",
    "fp_pass": "FP_pass",
    "rdf_nrmsd": "RDF_nRMSD",
    "rdf_pass": "RDF_pass",
    "spearman_rho": "Spearman_rho",
    "spearman_pass": "Spearman_pass",
    "passed": "all_criteria_pass",
}


def transitions_table(stabilities: Sequence[StabilityResult], block_times: Sequence[tuple[float, float]],
                      n_stable: int) -> pd.DataFrame:
    """One row per transition: the columns of convergence_metrics.csv.

    ``block_times[i]`` is the (start_ns, end_ns) of block i.
    """
    first = first_convergence(stabilities, n_stable)
    block = np.arange(1, len(stabilities) + 1)  # the later block of each transition
    start_ns, end_ns = np.array(block_times, dtype=float)[block].T
    blocks = pd.DataFrame({
        "transition": [f"{b - 1}->{b}" for b in block],
        "previous_block": block - 1,
        "block": block,
        "block_start_ns": start_ns,
        "block_end_ns": end_ns,
    })
    metrics = pd.DataFrame([asdict(s) for s in stabilities], columns=list(CSV_COLUMNS)).rename(columns=CSV_COLUMNS)
    return blocks.join(metrics).assign(
        stable_transition_count=stable_in_a_row(stabilities),
        convergence_reached=False if first is None else block - 1 >= first,
    )
