"""
Block-wise WaterFP for the candidate atoms and the convergence verdict.

Reuses the tracked ligand-side implementation unchanged:

* RDF: ``ligand_waterfp.waterfp.calculate_rdf.make_bins`` /
  ``compute_density_profile`` (water-O number density, 0-2.001 nm, 0.001 nm bins).
* FP: ``ligand_waterfp.waterfp.calculate_fingerprint.fp_from_density_profile``.
* Stability of one block transition:
  ``ligand_waterfp.convergence.monitor_convergence.check_stability`` with
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

import numpy as np
import pandas as pd
from ligand_waterfp.convergence.monitor_convergence import Tolerances, check_stability
from ligand_waterfp.waterfp.calculate_fingerprint import NORM_TAIL_BINS_DEFAULT, fp_from_density_profile
from waterfp import make_bins, compute_density_profile
from ligand_waterfp.waterfp.calculate_rdf import (
    RDF_BINWIDTH_NM_DEFAULT,
    RDF_RMAX_NM_DEFAULT
)

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
    _prev: tuple | None = None

    def add_block(self, block: int, t_start_ns: float, t_end_ns: float, fp: dict, g: dict,
                  order: list[str]) -> None:
        rank = fp_ranks(fp, order)
        if self._prev is not None:
            prev_fp, prev_g, prev_rank = self._prev
            res = check_stability(fp, g, rank, prev_fp, prev_g, prev_rank, self.tol)
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
        self._prev = (fp, g, rank)

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


def block_waterfp(u, start: int, end: int, cand_idx, water_o_idx, labels: list[str],
                  rmax_nm: float = RDF_RMAX_NM_DEFAULT, binwidth_nm: float = RDF_BINWIDTH_NM_DEFAULT,
                  norm_tail_bins: int = NORM_TAIL_BINS_DEFAULT):
    """FP and g(r) of each candidate over frames [start, end) (ligand-side WaterFP code)."""
    _edges_nm, centers_nm, edges_a, shell_vol = make_bins(rmax_nm, binwidth_nm)
    n_r = compute_density_profile(u, np.asarray(cand_idx), np.asarray(water_o_idx), start, end, edges_a, shell_vol)
    fp, g = {}, {}
    for i, name in enumerate(labels):
        value, g_i, _norm = fp_from_density_profile(n_r[i], centers_nm, binwidth_nm, norm_tail_bins)
        fp[name] = float(value)
        g[name] = g_i
    return fp, g, centers_nm
