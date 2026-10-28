"""End-to-end runs on synthetic systems (no funnel, no Zone, no PLUMED)."""

import json
import sys

import numpy as np
import pandas as pd
import pytest

from conftest import build_universe, simple_protein
from protein_waterfp import cli
from protein_waterfp.cli import AnalysisOptions, main, run_analysis
from protein_waterfp.final_table import FINAL_COLUMNS
from protein_waterfp.selection import CANDIDATE_CUTOFF_A_DEFAULT, select_candidates, select_system

LIGAND = [("C1", "C", (30.0, 16.0, 10.0)), ("N1", "N", (31.3, 16.0, 10.0)),
          ("O1", "O", (32.0, 17.1, 10.0)), ("H1", "H", (29.5, 16.8, 10.0))]
N_FRAMES, DT_PS, BLOCK_NS = 12, 1000.0, 2.0


def _system():
    return build_universe(simple_protein(), LIGAND, n_waters=150)


def _trajectory(u, n_frames=N_FRAMES, noise=0.3, dt_ps=DT_PS, t0_ps=0.0):
    from MDAnalysis.coordinates.memory import MemoryReader

    rng = np.random.default_rng(5)
    base = u.atoms.positions.copy()
    coords = np.array([base + rng.normal(0, noise, base.shape) for _ in range(n_frames)], dtype=np.float32)
    u.load_new(coords, format=MemoryReader, dt=dt_ps, dimensions=u.dimensions)
    if t0_ps:  # trajectory not starting at t = 0 (e.g. a continuation run)
        u.trajectory.ts.data["time_offset"] = t0_ps
    return u


def _run(tmp_path, **traj_kw):
    u_complex = _system()
    u_traj = _trajectory(_system(), **traj_kw)
    meta = run_analysis(u_complex, u_traj, AnalysisOptions(block_ns=BLOCK_NS), tmp_path, log=lambda *_: None)
    return u_traj, meta


@pytest.fixture
def analysis(tmp_path):
    u_complex = _system()
    u_traj = _trajectory(_system())
    meta = run_analysis(u_complex, u_traj, AnalysisOptions(block_ns=BLOCK_NS), tmp_path, log=lambda *_: None)
    return u_complex, meta, tmp_path


def test_no_zone_funnel_or_plumed_anywhere_in_a_run(analysis):
    _u, meta, out = analysis
    assert "plumed" not in sys.modules and not any(m.startswith("oneopes_funnel") for m in sys.modules)
    text = json.dumps(meta).lower()
    assert not any(word in text for word in ("zone", "funnel", "projection", "deep", "entrance", "plumed"))
    assert not list(out.glob("*funnel*")) and not list(out.glob("*zone*"))
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    assert "Zone" not in table.columns and "Funnel_axis_projection_mean_nm" not in table.columns


def test_cli_has_no_funnel_zone_or_plumed_options():
    base = ["--complex", "c", "--topology", "t", "--trajectory", "x", "--output-dir", "o"]
    cli.parse_args(base)
    for extra in (["--plumed", "p.dat"], ["--funnel-structure", "a.top"], ["--deep-end", "p0"],
                  ["--deep-middle-tol-nm", "0.0"]):
        with pytest.raises(SystemExit):
            cli.parse_args(base + extra)


def test_final_table_has_every_candidate_once(analysis):
    u, meta, out = analysis
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    expected = select_candidates(u, select_system(u), CANDIDATE_CUTOFF_A_DEFAULT)
    assert list(table.columns) == FINAL_COLUMNS
    assert len(table) == len(expected) == meta["n_candidate_atoms"]
    assert set(table["Atom_index"]) == set(expected["Atom_index"])
    assert list(table["FP_Rank"]) == list(range(1, len(table) + 1)) and table["FP"].is_monotonic_increasing
    assert set(table["Analysis_time_ns"]) == {meta["convergence"]["final_analysis_time_ns"]}


def test_final_values_cover_the_full_trajectory_not_one_block(tmp_path):
    from protein_waterfp.trajectory import structural_block
    from waterfp import WaterFP

    n_frames = 13  # 6 blocks of 2 frames + 1 trailing frame
    u, meta = _run(tmp_path, n_frames=n_frames)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv").set_index("Atom")
    sel = select_system(u)
    labels = list(pd.read_csv(tmp_path / "candidate_atom_mapping.csv")["Atom"])
    idx = table.loc[labels, "Atom_index"].to_numpy()
    fp_full = WaterFP(u.atoms[idx], u.atoms[sel.water_oxygens]).run(start=0, stop=n_frames).results.fp
    st_full = structural_block(u, 0, n_frames, idx, sel.ligand_heavy, sel.water_oxygens)
    np.testing.assert_allclose(table.loc[labels, "FP"], fp_full, rtol=1e-12)
    np.testing.assert_allclose(table.loc[labels, "Contact_occupancy_5A"], st_full.contact_occupancy)
    np.testing.assert_allclose(table.loc[labels, "Bridge_occupancy"], st_full.bridge_occupancy)
    last_block = pd.read_csv(tmp_path / "fp_values_per_block.csv").query("block == block.max()").set_index("atom")
    assert not np.allclose(table.loc[labels, "FP"], last_block.loc[labels, "FP"])  # not the last block
    assert set(table["Analysis_time_ns"]) == {(n_frames - 1) * DT_PS / 1000.0}


@pytest.mark.parametrize("n_frames, dt_ps, t0_ps", [(7, 1000.0, 0.0), (12, 1000.0, 0.0), (17, 1000.0, 0.0),
                                                     (31, 500.0, 0.0), (10, 1000.0, 40000.0)])
def test_arbitrary_lengths_blocks_and_transitions_derived_from_trajectory(tmp_path, n_frames, dt_ps, t0_ps):
    _u, meta = _run(tmp_path, n_frames=n_frames, dt_ps=dt_ps, t0_ps=t0_ps)
    fpb = int(round(BLOCK_NS * 1000 / dt_ps))
    n_blocks = n_frames // fpb
    summary = json.loads((tmp_path / "convergence_summary.json").read_text())
    assert summary["trajectory_length_ns"] == pytest.approx((n_frames - 1) * dt_ps / 1000.0)
    assert summary["final_analysis_time_ns"] == pytest.approx((n_frames - 1) * dt_ps / 1000.0)
    assert summary["trajectory_start_ns"] == pytest.approx(t0_ps / 1000.0)
    assert summary["number_of_blocks"] == n_blocks and summary["number_of_transitions"] == n_blocks - 1
    assert summary["frames_not_in_a_complete_block"] == n_frames - n_blocks * fpb
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert len(metrics) == n_blocks - 1
    expected_ends = [t0_ps / 1000.0 + BLOCK_NS * (b + 1) for b in range(1, n_blocks)]
    np.testing.assert_allclose(metrics["block_end_ns"], expected_ends)
    np.testing.assert_allclose(metrics["block_start_ns"], np.array(expected_ends) - BLOCK_NS)


def test_convergence_reported_retrospectively_and_analysis_continues(tmp_path):
    # Near-static trajectory: every transition is stable, so sustained convergence is reached
    # at the end of block 3 (third stable transition); all later blocks must still be analysed.
    n_frames = 20  # 10 blocks
    _u, meta = _run(tmp_path, n_frames=n_frames, noise=0.0)
    summary = json.loads((tmp_path / "convergence_summary.json").read_text())
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert metrics["all_criteria_pass"].all()
    assert summary["convergence_reached"] is True
    assert summary["first_sustained_convergence_ns"] == pytest.approx(4 * BLOCK_NS)
    assert summary["first_sustained_convergence_transition"] == "2->3"
    assert summary["number_of_blocks"] == 10 and len(metrics) == 9  # nothing truncated
    assert summary["n_transitions_after_first_convergence"] == 6
    assert summary["remained_stable_after_first_convergence"] is True
    assert pd.read_csv(tmp_path / "fp_values_per_block.csv")["block"].max() == 9
    assert list(metrics["convergence_reached"]) == [False, False] + [True] * 7
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv")
    assert set(table["First_convergence_ns"]) == {4 * BLOCK_NS}
    assert set(table["Analysis_time_ns"]) == {(n_frames - 1) * DT_PS / 1000.0}


def test_convergence_metrics_are_a_separate_table(analysis):
    _u, _meta, out = analysis
    metrics = pd.read_csv(out / "convergence_metrics.csv")
    for col in ("block_start_ns", "block_end_ns", "FP_relative_change", "FP_pass", "RDF_nRMSD", "RDF_pass",
                "Spearman_rho", "Spearman_pass", "all_criteria_pass", "stable_transition_count",
                "convergence_reached"):
        assert col in metrics.columns, col
    table = pd.read_csv(out / "final_per_atom_table.csv")
    assert not set(metrics.columns) & set(table.columns)


def test_worst_case_metrics_match_per_atom_block_data(analysis):
    from ligand_waterfp.convergence.monitor_convergence import nrmsd

    _u, _meta, out = analysis
    fp = pd.read_csv(out / "fp_values_per_block.csv")
    rdf = pd.read_csv(out / "rdf_profiles_per_block.csv")
    metrics = pd.read_csv(out / "convergence_metrics.csv").set_index("block")
    last, prev = fp["block"].max(), fp["block"].max() - 1
    f_last = fp[fp.block == last].set_index("atom")["FP"]
    f_prev = fp[fp.block == prev].set_index("atom")["FP"]
    rel = (f_last - f_prev).abs() / f_prev.abs().clip(lower=1e-6)
    assert metrics.loc[last, "FP_relative_change"] == pytest.approx(rel.max(), rel=1e-9)
    g = {b: rdf[rdf.block == b].sort_values(["atom", "r_nm"]).groupby("atom")["g_r"].apply(np.asarray)
         for b in (prev, last)}
    worst = max(nrmsd(g[last][a], g[prev][a]) for a in g[last].index)
    assert metrics.loc[last, "RDF_nRMSD"] == pytest.approx(worst, rel=1e-6)


def test_block_data_kept_for_every_block_and_atom(analysis):
    _u, meta, out = analysis
    n_blocks = int(N_FRAMES * DT_PS // (BLOCK_NS * 1000))
    n = meta["n_candidate_atoms"]
    fp = pd.read_csv(out / "fp_values_per_block.csv")
    assert fp["block"].nunique() == n_blocks and len(fp) == n_blocks * n
    assert len(pd.read_csv(out / "contact_occupancy_per_block.csv")) == n_blocks * n
    assert len(pd.read_csv(out / "convergence_metrics.csv")) == n_blocks - 1
    assert meta["convergence"]["number_of_blocks"] == n_blocks
    assert len(pd.read_csv(out / "rdf_profiles_per_block.csv")) == n_blocks * n * 2001
    ranks = pd.read_csv(out / "fp_ranking_by_block_wide.csv", index_col=0)
    values = pd.read_csv(out / "fp_values_by_block_wide.csv", index_col=0)
    last = values.columns[-1]
    assert ranks[last].idxmin() == values[last].idxmin()  # rank 1 = most negative in every file
    summary = json.loads((out / "convergence_summary.json").read_text())
    for key in ("trajectory_length_ns", "block_size_ns", "number_of_blocks", "number_of_transitions",
                "stable_transitions_required", "fp_threshold", "rdf_nrmsd_threshold", "spearman_threshold",
                "first_sustained_convergence_ns", "final_analysis_time_ns", "convergence_reached"):
        assert key in summary, key
    assert (summary["fp_threshold"], summary["rdf_nrmsd_threshold"], summary["spearman_threshold"]) == (0.10, 0.15, 0.90)
    assert summary["stable_transitions_required"] == 3


def test_cli_end_to_end_from_files(tmp_path):
    import MDAnalysis as mda

    u = _trajectory(_system())
    work = tmp_path / "case"
    work.mkdir()
    u.atoms.write(work / "sys.pdb")
    with mda.Writer(str(work / "traj.xtc"), len(u.atoms)) as w:
        for _ts in u.trajectory:
            w.write(u.atoms)
    out = tmp_path / "result"
    rc = main(["--complex", str(work / "sys.pdb"), "--topology", str(work / "sys.pdb"),
               "--trajectory", str(work / "traj.xtc"), "--output-dir", str(out), "--block-ns", str(BLOCK_NS)])
    assert rc == 0
    for name in ("candidate_atom_mapping.csv", "fp_values_per_block.csv", "rdf_profiles_per_block.csv",
                 "fp_values_by_block_wide.csv", "fp_ranking_by_block_wide.csv", "convergence_metrics.csv",
                 "convergence_summary.json", "convergence_summary.txt", "final_per_atom_table.csv",
                 "final_per_atom_table.md",
                 "run_metadata.json"):
        assert (out / name).exists(), name
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    assert set(table["Chain_ID"]) == {"B"}  # the chain written in the complex file
    meta_text = (out / "run_metadata.json").read_text()
    assert str(tmp_path) not in meta_text
    assert json.loads(meta_text)["candidate_cutoff_A"] == 7.0  # default


@pytest.mark.parametrize("cutoff", [7.0, 10.0])
def test_explicit_cutoff_selects_exactly_that_pool(tmp_path, cutoff):
    u_complex = _system()
    u_traj = _trajectory(_system(), n_frames=4)
    meta = run_analysis(u_complex, u_traj, AnalysisOptions(candidate_cutoff_A=cutoff, block_ns=BLOCK_NS), tmp_path,
                        log=lambda *_: None)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv")
    expected = select_candidates(u_complex, select_system(u_complex), cutoff)
    assert meta["candidate_cutoff_A"] == cutoff
    assert set(table["Atom_index"]) == set(expected["Atom_index"]) and len(table) == len(expected)
    assert table["Min_distance_to_ligand_A"].max() <= cutoff


def test_10_A_pool_is_a_superset_of_the_7_A_pool(small_system):
    sel = select_system(small_system)
    a7 = set(select_candidates(small_system, sel, 7.0)["Atom_index"])
    a10 = set(select_candidates(small_system, sel, 10.0)["Atom_index"])
    assert a7 < a10


def test_cli_explicit_10_A_cutoff_end_to_end(tmp_path):
    import MDAnalysis as mda

    u = _trajectory(_system(), n_frames=4)
    work = tmp_path / "case"
    work.mkdir()
    u.atoms.write(work / "sys.pdb")
    with mda.Writer(str(work / "traj.xtc"), len(u.atoms)) as w:
        for _ts in u.trajectory:
            w.write(u.atoms)
    out = tmp_path / "result"
    assert main(["--complex", str(work / "sys.pdb"), "--topology", str(work / "sys.pdb"),
                 "--trajectory", str(work / "traj.xtc"), "--output-dir", str(out), "--block-ns", str(BLOCK_NS),
                 "--candidate-cutoff-A", "10.0"]) == 0
    assert json.loads((out / "run_metadata.json").read_text())["candidate_cutoff_A"] == 10.0
    assert pd.read_csv(out / "final_per_atom_table.csv")["Min_distance_to_ligand_A"].max() <= 10.0


def test_missing_input_fails_early(tmp_path):
    with pytest.raises(SystemExit, match="not found"):
        main(["--complex", str(tmp_path / "nope.pdb"), "--topology", "x", "--trajectory", "y",
              "--output-dir", str(tmp_path / "o")])
