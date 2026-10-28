"""
Unit tests for the pure/reusable pieces extracted from
ligand_waterfp.convergence.monitor_convergence's main() (code-review
Rules 2, 3 and 5) - check_stability() (now returning a frozen
StabilityResult dataclass instead of a dict), the consecutive-streak
requirement built from it, write_reports() (now taking a Tolerances
dataclass), and plot_atom(). All synthetic data, no real
trajectory/MDAnalysis needed - the per-block waterfp.WaterFP analysis in
main() needs a real .tpr/.xtc and is covered by tests/test_waterfp.py.

check_stability() takes per-atom arrays: fp of shape (n_atoms,) and g of
shape (n_atoms, n_bins), atoms in the same order in both blocks.

Tolerances used below (fp_rel_tol=0.10, rdf_nrmsd_tol=0.15,
spearman_tol=0.90, n_stable=3) are the actual defaults from
monitor_convergence.parse_args() / configs/ligand_waterfp.yaml - not
invented values.
"""
import dataclasses

import numpy as np
import pandas as pd
import pytest

from ligand_waterfp.convergence.monitor_convergence import (
    check_stability,
    write_reports,
    plot_atom,
    StabilityResult,
    Tolerances,
)

FP_REL_TOL = 0.10
RDF_NRMSD_TOL = 0.15
SPEARMAN_TOL = 0.90
N_STABLE = 3
TOL = Tolerances(
    fp_rel_tol=FP_REL_TOL,
    rdf_nrmsd_tol=RDF_NRMSD_TOL,
    spearman_tol=SPEARMAN_TOL,
    n_stable=N_STABLE,
)


def _same_g(n_atoms):
    """Identical g(r) for every atom (and, reused across blocks, nRMSD = 0)."""
    return np.tile([0.0, 0.5, 1.0, 1.0, 1.0], (n_atoms, 1))


def test_stable_transition_passes_all_three_criteria():
    prev_fp = np.array([-10.0, -8.0, -5.0])
    fp_this = np.array([-10.5, -8.3, -5.1])  # 5%, 3.75%, 2% changes - same order
    g = _same_g(3)

    result = check_stability(fp_this, g, prev_fp, g, TOL)

    assert isinstance(result, StabilityResult)
    assert result.value_stable
    assert result.rdf_stable
    assert result.rank_stable
    assert result.block_stable
    assert result.max_fp_relative_change == pytest.approx(0.05)
    assert result.max_rdf_nrmsd == pytest.approx(0.0)
    assert result.spearman_rank_corr == pytest.approx(1.0)


def test_fp_change_above_threshold_makes_transition_unstable():
    prev_fp = np.array([-10.0, -8.0, -5.0])
    fp_this = np.array([-11.5, -8.1, -5.05])  # first atom changed 15% > 10% tol; order unchanged
    g = _same_g(3)

    result = check_stability(fp_this, g, prev_fp, g, TOL)

    assert not result.value_stable
    assert result.rdf_stable     # isolate: only FP failed
    assert result.rank_stable
    assert not result.block_stable
    assert result.max_fp_relative_change == pytest.approx(0.15)


def test_rdf_nrmsd_above_threshold_makes_transition_unstable():
    prev_fp = np.array([-10.0, -8.0, -5.0])
    fp_this = np.array([-10.2, -8.1, -5.05])  # small changes, order unchanged
    prev_g = _same_g(3)
    # the first atom's g(r) shape changes a lot between blocks -> large nRMSD
    g_this = prev_g.copy()
    g_this[0] = [1.0, 1.0, 1.0, 0.5, 0.0]

    result = check_stability(fp_this, g_this, prev_fp, prev_g, TOL)

    assert result.value_stable
    assert not result.rdf_stable    # isolate: only RDF failed
    assert result.rank_stable
    assert not result.block_stable
    assert result.max_rdf_nrmsd > RDF_NRMSD_TOL


def test_ranking_instability_makes_transition_unstable():
    # Closely-spaced FP values so small (<=10%) per-atom wiggles can still
    # fully reverse the relative order between blocks.
    prev_fp = np.array([-10.0, -10.1, -10.2, -10.3])
    fp_this = np.array([-10.25, -10.15, -10.05, -9.95])  # order reversed
    g = _same_g(4)

    result = check_stability(fp_this, g, prev_fp, g, TOL)

    # sanity: every individual atom's own relative change stayed under 10%
    assert np.all(np.abs(fp_this - prev_fp) / np.abs(prev_fp) <= FP_REL_TOL)

    assert result.value_stable
    assert result.rdf_stable
    assert not result.rank_stable   # isolate: only ranking failed
    assert not result.block_stable
    assert result.spearman_rank_corr == pytest.approx(-1.0)  # full reversal


def test_requires_n_stable_consecutive_transitions():
    """Replicates main()'s own streak bookkeeping
    (stable_streak = stable_streak + 1 if block_stable else 0;
    converged once stable_streak >= n_stable) around check_stability(),
    using N_STABLE=3 - the actual configured value, not an invented one.
    A single unstable transition partway through must reset the streak,
    so convergence is only reached after a genuinely fresh run of 3
    consecutive stable transitions."""
    g = _same_g(2)

    # block FP values: 0,1,2 stable; 2->3 unstable (big jump); 3,4,5,6 stable again
    blocks_fp = np.array([
        [-10.0, -5.0],   # block 0 (baseline)
        [-10.1, -5.05],  # block 1: stable vs 0
        [-10.2, -5.1],   # block 2: stable vs 1
        [-14.0, -5.15],  # block 3: UNSTABLE vs 2 (first atom changed >10%)
        [-14.1, -5.2],   # block 4: stable vs 3 -> streak resets to 1
        [-14.2, -5.25],  # block 5: stable vs 4 -> streak 2
        [-14.3, -5.3],   # block 6: stable vs 5 -> streak 3 -> CONVERGED here
    ])

    stable_streak = 0
    streak_history = []
    converged_at = None
    prev_fp = blocks_fp[0]

    for block_idx in range(1, len(blocks_fp)):
        fp_this = blocks_fp[block_idx]
        result = check_stability(fp_this, g, prev_fp, g, TOL)
        stable_streak = stable_streak + 1 if result.block_stable else 0
        streak_history.append(stable_streak)
        if converged_at is None and stable_streak >= N_STABLE:
            converged_at = block_idx
        prev_fp = fp_this

    # transitions 0->1, 1->2 stable (streak 1, 2); 2->3 unstable (streak resets to 0);
    # 3->4, 4->5 stable (streak 1, 2); 5->6 stable (streak 3 -> converged at block 6)
    assert streak_history == [1, 2, 0, 1, 2, 3]
    assert converged_at == 6


def test_stability_result_matches_previous_dict_implementation_values():
    """check_stability() used to return a dict with these exact 7
    key/value pairs (PR2); this pins the StabilityResult dataclass (PR3)
    to produce the identical numerical/boolean values for the same
    inputs - the refactor changed the container, not the computation."""
    prev_fp = np.array([-10.0, -8.0, -5.0])
    fp_this = np.array([-10.5, -8.3, -5.1])
    g = _same_g(3)

    result = check_stability(fp_this, g, prev_fp, g, TOL)

    expected = {
        "max_fp_relative_change": pytest.approx(0.05),
        "max_rdf_nrmsd": pytest.approx(0.0),
        "spearman_rank_corr": pytest.approx(1.0),
        "value_stable": True,
        "rdf_stable": True,
        "rank_stable": True,
        "block_stable": True,
    }
    for field, expected_value in expected.items():
        assert getattr(result, field) == expected_value


def test_stability_result_is_frozen():
    fp = np.array([-10.0, -5.0])
    g = _same_g(2)
    result = check_stability(fp, g, fp, g, TOL)
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.block_stable = False


def test_tolerances_dataclass_reproduces_configured_defaults_behavior():
    """Constructing Tolerances explicitly from the documented config
    defaults (fp_rel_tol=0.10, rdf_nrmsd_tol=0.15, spearman_tol=0.90,
    n_stable=3) and running it through check_stability() must behave
    identically to passing the 4 values independently (the pre-PR3 call
    convention) - same stability verdict, same threshold comparisons."""
    tol = Tolerances(fp_rel_tol=0.10, rdf_nrmsd_tol=0.15, spearman_tol=0.90, n_stable=3)
    assert tol.fp_rel_tol == FP_REL_TOL
    assert tol.rdf_nrmsd_tol == RDF_NRMSD_TOL
    assert tol.spearman_tol == SPEARMAN_TOL
    assert tol.n_stable == N_STABLE

    prev_fp = np.array([-10.0, -8.0])
    fp_this = np.array([-11.5, -8.1])  # first atom changed 15% > 10% tol
    g = _same_g(2)

    result = check_stability(fp_this, g, prev_fp, g, tol)
    assert not result.value_stable
    assert not result.block_stable


def test_tolerances_is_frozen():
    tol = Tolerances(fp_rel_tol=0.10, rdf_nrmsd_tol=0.15, spearman_tol=0.90, n_stable=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        tol.n_stable = 5


def test_write_reports_preserves_schema_and_field_values(tmp_path):
    fp_df = pd.DataFrame([
        {"block": 0, "atom": "A", "FP": -10.0},
        {"block": 1, "atom": "A", "FP": -10.1},
        {"block": 0, "atom": "B", "FP": -5.0},
        {"block": 1, "atom": "B", "FP": -5.1},
    ])

    summary = write_reports(
        outdir=str(tmp_path), fp_df=fp_df, converged=True, stop_reason="test reason",
        stop_block=1, stop_time_ns=10.0, block_ns=5.0, tol=TOL,
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

    assert fp_plot_path == tmp_path / "fp_plots" / "FP_vs_time_A.png"
    assert rdf_plot_path == tmp_path / "rdf_plots" / "RDF_per_block_A.png"
    assert (tmp_path / "fp_plots" / "FP_vs_time_A.png").exists()
    assert (tmp_path / "rdf_plots" / "RDF_per_block_A.png").exists()
