"""
Unit tests for the pure/reusable pieces extracted from
ligand_waterfp.convergence.monitor_convergence's main() (code-review
Rules 2 and 5) - check_stability(), the consecutive-streak requirement
built from it, write_reports(), and plot_atom(). All synthetic data, no
real trajectory/MDAnalysis needed - analyze_block() itself still needs a
real .tpr/.xtc (it calls waterfp.calculate_rdf.compute_density_profile(),
which reads real trajectory frames) and is intentionally NOT covered here,
same as this project's existing precedent for the other
trajectory-dependent scripts (see tests/README.md).

Tolerances used below (fp_rel_tol=0.10, rdf_nrmsd_tol=0.15,
spearman_tol=0.90, n_stable=3) are the actual defaults from
monitor_convergence.parse_args() / configs/ligand_waterfp.yaml - not
invented values.
"""
import numpy as np
import pandas as pd
import pytest

from ligand_waterfp.convergence.monitor_convergence import (
    check_stability,
    write_reports,
    plot_atom,
)

FP_REL_TOL = 0.10
RDF_NRMSD_TOL = 0.15
SPEARMAN_TOL = 0.90
N_STABLE = 3


def _rank(fp_dict):
    return pd.Series(fp_dict).rank(ascending=False)


def test_stable_transition_passes_all_three_criteria():
    prev_fp = {"A": -10.0, "B": -8.0, "C": -5.0}
    fp_this = {"A": -10.5, "B": -8.3, "C": -5.1}  # 5%, 3.75%, 2% changes - same order
    g = {name: np.array([0.0, 0.5, 1.0, 1.0, 1.0]) for name in prev_fp}  # identical g(r) -> nRMSD = 0

    result = check_stability(fp_this, g, _rank(fp_this), prev_fp, g, _rank(prev_fp),
                              FP_REL_TOL, RDF_NRMSD_TOL, SPEARMAN_TOL)

    assert result["value_stable"]
    assert result["rdf_stable"]
    assert result["rank_stable"]
    assert result["block_stable"]
    assert result["max_FP_relative_change"] == pytest.approx(0.05)
    assert result["max_RDF_nRMSD"] == pytest.approx(0.0)
    assert result["spearman_rank_corr"] == pytest.approx(1.0)


def test_fp_change_above_threshold_makes_transition_unstable():
    prev_fp = {"A": -10.0, "B": -8.0, "C": -5.0}
    fp_this = {"A": -11.5, "B": -8.1, "C": -5.05}  # A changed 15% > 10% tol; order unchanged
    g = {name: np.array([0.0, 0.5, 1.0, 1.0, 1.0]) for name in prev_fp}

    result = check_stability(fp_this, g, _rank(fp_this), prev_fp, g, _rank(prev_fp),
                              FP_REL_TOL, RDF_NRMSD_TOL, SPEARMAN_TOL)

    assert not result["value_stable"]
    assert result["rdf_stable"]     # isolate: only FP failed
    assert result["rank_stable"]
    assert not result["block_stable"]
    assert result["max_FP_relative_change"] == pytest.approx(0.15)


def test_rdf_nrmsd_above_threshold_makes_transition_unstable():
    prev_fp = {"A": -10.0, "B": -8.0, "C": -5.0}
    fp_this = {"A": -10.2, "B": -8.1, "C": -5.05}  # small changes, order unchanged
    prev_g = {name: np.array([0.0, 0.5, 1.0, 1.0, 1.0]) for name in prev_fp}
    # atom A's g(r) shape changes a lot between blocks -> large nRMSD
    g_this = dict(prev_g)
    g_this["A"] = np.array([1.0, 1.0, 1.0, 0.5, 0.0])

    result = check_stability(fp_this, g_this, _rank(fp_this), prev_fp, prev_g, _rank(prev_fp),
                              FP_REL_TOL, RDF_NRMSD_TOL, SPEARMAN_TOL)

    assert result["value_stable"]
    assert not result["rdf_stable"]    # isolate: only RDF failed
    assert result["rank_stable"]
    assert not result["block_stable"]
    assert result["max_RDF_nRMSD"] > RDF_NRMSD_TOL


def test_ranking_instability_makes_transition_unstable():
    # Closely-spaced FP values so small (<=10%) per-atom wiggles can still
    # fully reverse the relative order between blocks.
    prev_fp = {"A": -10.0, "B": -10.1, "C": -10.2, "D": -10.3}
    fp_this = {"A": -10.25, "B": -10.15, "C": -10.05, "D": -9.95}  # order reversed
    g = {name: np.array([0.0, 0.5, 1.0, 1.0, 1.0]) for name in prev_fp}  # identical -> nRMSD=0

    prev_rank = _rank(prev_fp)
    rank_now = _rank(fp_this)
    result = check_stability(fp_this, g, rank_now, prev_fp, g, prev_rank,
                              FP_REL_TOL, RDF_NRMSD_TOL, SPEARMAN_TOL)

    # sanity: every individual atom's own relative change stayed under 10%
    for name in prev_fp:
        assert abs(fp_this[name] - prev_fp[name]) / abs(prev_fp[name]) <= FP_REL_TOL

    assert result["value_stable"]
    assert result["rdf_stable"]
    assert not result["rank_stable"]   # isolate: only ranking failed
    assert not result["block_stable"]
    assert result["spearman_rank_corr"] == pytest.approx(-1.0)  # full reversal


def test_requires_n_stable_consecutive_transitions():
    """Replicates main()'s own streak bookkeeping
    (stable_streak = stable_streak + 1 if block_stable else 0;
    converged once stable_streak >= n_stable) around check_stability(),
    using N_STABLE=3 - the actual configured value, not an invented one.
    A single unstable transition partway through must reset the streak,
    so convergence is only reached after a genuinely fresh run of 3
    consecutive stable transitions."""
    g_stable = np.array([0.0, 0.5, 1.0, 1.0, 1.0])
    g = {"A": g_stable, "B": g_stable}

    # block FP values: 0,1,2 stable; 2->3 unstable (big jump); 3,4,5,6 stable again
    blocks_fp = [
        {"A": -10.0, "B": -5.0},   # block 0 (baseline)
        {"A": -10.1, "B": -5.05},  # block 1: stable vs 0
        {"A": -10.2, "B": -5.1},   # block 2: stable vs 1
        {"A": -14.0, "B": -5.15},  # block 3: UNSTABLE vs 2 (A changed >10%)
        {"A": -14.1, "B": -5.2},   # block 4: stable vs 3 -> streak resets to 1
        {"A": -14.2, "B": -5.25},  # block 5: stable vs 4 -> streak 2
        {"A": -14.3, "B": -5.3},   # block 6: stable vs 5 -> streak 3 -> CONVERGED here
    ]

    stable_streak = 0
    streak_history = []
    converged_at = None
    prev_fp = blocks_fp[0]
    prev_rank = _rank(prev_fp)

    for block_idx in range(1, len(blocks_fp)):
        fp_this = blocks_fp[block_idx]
        rank_now = _rank(fp_this)
        result = check_stability(fp_this, g, rank_now, prev_fp, g, prev_rank,
                                  FP_REL_TOL, RDF_NRMSD_TOL, SPEARMAN_TOL)
        stable_streak = stable_streak + 1 if result["block_stable"] else 0
        streak_history.append(stable_streak)
        if converged_at is None and stable_streak >= N_STABLE:
            converged_at = block_idx
        prev_fp, prev_rank = fp_this, rank_now

    # transitions 0->1, 1->2 stable (streak 1, 2); 2->3 unstable (streak resets to 0);
    # 3->4, 4->5 stable (streak 1, 2); 5->6 stable (streak 3 -> converged at block 6)
    assert streak_history == [1, 2, 0, 1, 2, 3]
    assert converged_at == 6


def test_write_reports_preserves_schema_and_field_values(tmp_path):
    fp_df = pd.DataFrame([
        {"block": 0, "atom": "A", "FP": -10.0},
        {"block": 1, "atom": "A", "FP": -10.1},
        {"block": 0, "atom": "B", "FP": -5.0},
        {"block": 1, "atom": "B", "FP": -5.1},
    ])

    summary = write_reports(
        outdir=str(tmp_path), fp_df=fp_df, converged=True, stop_reason="test reason",
        stop_block=1, stop_time_ns=10.0, block_ns=5.0, n_stable=N_STABLE,
        fp_rel_tol=FP_REL_TOL, rdf_nrmsd_tol=RDF_NRMSD_TOL, spearman_tol=SPEARMAN_TOL,
        ligand_resname="MOL", heavy_names=["A", "B"], n_blocks_completed=2,
    )

    assert summary["converged"] is True
    assert summary["stop_block"] == 1
    assert summary["n_stable_transitions_required"] == N_STABLE
    assert summary["thresholds"]["FP_relative_change_tol"] == FP_REL_TOL
    assert summary["ligand_heavy_atoms"] == ["A", "B"]

    json_path = tmp_path / "convergence_summary.json"
    txt_path = tmp_path / "convergence_summary.txt"
    wide_fp_path = tmp_path / "fp_values_by_block_wide.csv"
    wide_rank_path = tmp_path / "fp_ranking_by_block_wide.csv"
    for p in (json_path, txt_path, wide_fp_path, wide_rank_path):
        assert p.exists()

    assert "Converged: True" in txt_path.read_text()
    wide_fp = pd.read_csv(wide_fp_path, index_col="atom")
    assert list(wide_fp.columns) == ["0", "1"]
    assert wide_fp.loc["A", "0"] == pytest.approx(-10.0)


def test_plot_atom_writes_expected_files(tmp_path):
    (tmp_path / "fp_plots").mkdir()
    (tmp_path / "rdf_plots").mkdir()

    fp_df = pd.DataFrame([
        {"t_mid_ns": 2.5, "atom": "A", "FP": -10.0},
        {"t_mid_ns": 7.5, "atom": "A", "FP": -10.1},
    ])
    rdf_df = pd.DataFrame([
        {"block": 0, "atom": "A", "r_nm": 0.1, "g_r": 0.0},
        {"block": 0, "atom": "A", "r_nm": 0.2, "g_r": 1.0},
        {"block": 1, "atom": "A", "r_nm": 0.1, "g_r": 0.0},
        {"block": 1, "atom": "A", "r_nm": 0.2, "g_r": 1.0},
    ])

    fp_plot_path, rdf_plot_path = plot_atom("A", fp_df, rdf_df, str(tmp_path))

    assert fp_plot_path == str(tmp_path / "fp_plots" / "FP_vs_time_A.png")
    assert rdf_plot_path == str(tmp_path / "rdf_plots" / "RDF_per_block_A.png")
    assert (tmp_path / "fp_plots" / "FP_vs_time_A.png").exists()
    assert (tmp_path / "rdf_plots" / "RDF_per_block_A.png").exists()
