"""Tests for the ligand convergence monitor's reports and plots (synthetic data, no trajectory)."""
import pandas as pd

from convergence import DEFAULT_TOLERANCES as TOL
from ligand_waterfp.convergence.monitor_convergence import plot_atom, write_reports


def test_write_reports_preserves_schema_and_field_values(tmp_path):
    summary = write_reports(
        outdir=str(tmp_path), converged=True, stop_reason="test reason",
        stop_block=1, stop_time_ns=10.0, block_ns=5.0, tol=TOL,
        ligand_resname="MOL", heavy_names=["A", "B"], n_blocks_completed=2,
    )

    assert summary["converged"] is True
    assert summary["stop_block"] == 1
    assert summary["n_stable_transitions_required"] == TOL.n_stable
    assert summary["thresholds"]["FP_relative_change_tol"] == TOL.fp_rel_tol
    assert summary["ligand_heavy_atoms"] == ["A", "B"]

    assert (tmp_path / "convergence_summary.json").exists()
    assert "Converged: True" in (tmp_path / "convergence_summary.txt").read_text()


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
