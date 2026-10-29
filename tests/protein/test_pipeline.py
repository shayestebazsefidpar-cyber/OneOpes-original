"""End-to-end runs on synthetic systems (no funnel, no Zone, no PLUMED)."""

import json

import numpy as np
import pandas as pd
import pytest

from cyclopts import ValidationError

from oneopes.block_run import BlockOptions, load_run, run_metadata
from oneopes.cli import app, main
from oneopes.protein.analysis import AnalysisOptions, analysis_details, run_analysis, write_outputs
from oneopes.provenance import to_json
from oneopes.protein.candidates import CANDIDATE_CUTOFF_A_DEFAULT, select_candidates
from oneopes.protein.final_table import FINAL_COLUMNS
from oneopes.protein.inputs import protein_system
from oneopes.system import select_system
from protein_systems import build_universe, protein_args, simple_protein

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


def _run(tmp_path, options=AnalysisOptions(blocks=BlockOptions(block_ns=BLOCK_NS)), **traj_kw):
    """Run the analysis on the synthetic system; return the trajectory universe and the run metadata."""
    u_traj = _trajectory(_system(), **traj_kw)
    analysis = run_analysis(protein_system(u_traj, _system()), options)
    write_outputs(tmp_path, analysis)
    return u_traj, analysis_details(analysis)


@pytest.fixture
def analysis(tmp_path):
    _u_traj, meta = _run(tmp_path)
    return _system(), meta, tmp_path


def test_no_zone_funnel_or_plumed_anywhere_in_a_run(analysis):
    _u, meta, out = analysis
    text = to_json(meta).lower()
    assert not any(word in text for word in ("zone", "funnel", "projection", "deep", "entrance", "plumed"))
    assert not list(out.glob("*funnel*")) and not list(out.glob("*zone*"))
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    assert "Zone" not in table.columns and "Funnel_axis_projection_mean_nm" not in table.columns


def test_cli_has_no_funnel_zone_or_plumed_options(tmp_path):
    base = protein_args(tmp_path)
    app.parse_args(base)
    for extra in (["--plumed", "p.dat"], ["--funnel-structure", "a.top"], ["--deep-end", "p0"],
                  ["--deep-middle-tol-nm", "0.0"]):
        with pytest.raises(SystemExit):
            app.parse_args(base + extra, print_error=False)


def test_final_table_has_every_candidate_once(analysis):
    u, meta, out = analysis
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    expected = select_candidates(protein_system(u, u), CANDIDATE_CUTOFF_A_DEFAULT)
    assert list(table.columns) == FINAL_COLUMNS
    assert len(table) == expected.sizes["atom"] == meta["n_candidate_atoms"]
    assert set(table["Atom_index"]) == set(expected.atom_index.values)
    assert list(table["FP_Rank"]) == list(range(1, len(table) + 1)) and table["FP"].is_monotonic_increasing


def test_final_values_cover_the_full_trajectory_not_one_block(tmp_path):
    from oneopes.protein.contacts import Contacts
    from oneopes.waterfp import measure_hydration

    n_frames = 13  # 6 blocks of 2 frames + 1 trailing frame
    u, _meta = _run(tmp_path, n_frames=n_frames)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv").set_index("Atom")
    sel = select_system(u)
    labels = list(pd.read_csv(tmp_path / "candidate_atom_mapping.csv")["Atom"])
    idx = table.loc[labels, "Atom_index"].to_numpy()
    fp_full = measure_hydration(u.atoms[idx], sel.water(u), range(n_frames)).fp
    contacts = Contacts(u.atoms[idx], sel.ligand(u), sel.water(u)).run().results
    np.testing.assert_allclose(table.loc[labels, "FP"], fp_full, rtol=1e-12)
    np.testing.assert_allclose(table.loc[labels, "Contact_occupancy_5A"], contacts.contact_occupancy)
    np.testing.assert_allclose(table.loc[labels, "Bridge_occupancy"], contacts.bridge_occupancy)
    last_block = pd.read_csv(tmp_path / "fp_values_per_block.csv").query("block == block.max()").set_index("atom")
    assert not np.allclose(table.loc[labels, "FP"], last_block.loc[labels, "FP"])  # not the last block


@pytest.mark.parametrize("n_frames, dt_ps, t0_ps", [(7, 1000.0, 0.0), (12, 1000.0, 0.0), (17, 1000.0, 0.0),
                                                     (31, 500.0, 0.0), (10, 1000.0, 40000.0)])
def test_arbitrary_lengths_blocks_and_transitions_derived_from_trajectory(tmp_path, n_frames, dt_ps, t0_ps):
    _u, meta = _run(tmp_path, n_frames=n_frames, dt_ps=dt_ps, t0_ps=t0_ps)
    fpb = round(BLOCK_NS * 1000 / dt_ps)
    n_blocks = n_frames // fpb
    run_ns = (n_frames - 1) * dt_ps / 1000.0
    summary = json.loads((tmp_path / "convergence_summary.json").read_text())
    assert summary["trajectory_start_ns"] == pytest.approx(t0_ps / 1000.0)
    assert summary["trajectory_end_ns"] - summary["trajectory_start_ns"] == pytest.approx(run_ns)
    assert summary["analysis_end_ns"] - summary["analysis_start_ns"] == pytest.approx(run_ns)
    assert summary["n_blocks"] == n_blocks
    assert meta["frames"]["analysed"].stop - meta["frames"]["blocks"][-1]["frames"].stop == n_frames - n_blocks * fpb
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert len(metrics) == n_blocks - 1
    expected_ends = [t0_ps / 1000.0 + BLOCK_NS * (b + 1) for b in range(1, n_blocks)]
    np.testing.assert_allclose(metrics["t_end_ns"], expected_ends)
    np.testing.assert_allclose(metrics["t_start_ns"], np.array(expected_ends) - BLOCK_NS)


def test_convergence_reported_retrospectively_and_analysis_continues(tmp_path):
    # Near-static trajectory: every transition is stable, so sustained convergence is reached
    # at the end of block 3 (third stable transition); all later blocks must still be analysed.
    n_frames = 20  # 10 blocks
    _run(tmp_path, n_frames=n_frames, noise=0.0)
    summary = json.loads((tmp_path / "convergence_summary.json").read_text())
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert metrics["passed"].all()
    assert summary["converged"] is True
    assert summary["converged_ns"] == pytest.approx(4 * BLOCK_NS)
    assert summary["converged_block"] == 3
    assert summary["n_blocks"] == 10 and len(metrics) == 9  # nothing truncated
    assert summary["unstable_after_convergence"] == 0
    assert pd.read_csv(tmp_path / "fp_values_per_block.csv")["block"].max() == 9
    assert list(metrics["stable_in_a_row"]) == list(range(1, 10))
    assert summary["analysis_end_ns"] - summary["analysis_start_ns"] == pytest.approx((n_frames - 1) * DT_PS / 1000.0)


def test_convergence_metrics_are_a_separate_table(analysis):
    _u, _meta, out = analysis
    metrics = pd.read_csv(out / "convergence_metrics.csv")
    for col in ("block", "t_start_ns", "t_end_ns", "fp_change", "fp_pass", "rdf_nrmsd", "rdf_pass",
                "spearman_rho", "spearman_pass", "passed", "stable_in_a_row"):
        assert col in metrics.columns, col
    table = pd.read_csv(out / "final_per_atom_table.csv")
    assert not set(metrics.columns) & set(table.columns)


def test_worst_case_metrics_match_per_atom_block_data(analysis):
    from oneopes.convergence import nrmsd

    _u, _meta, out = analysis
    fp = pd.read_csv(out / "fp_values_per_block.csv")
    rdf = pd.read_csv(out / "rdf_profiles_per_block.csv")
    metrics = pd.read_csv(out / "convergence_metrics.csv").set_index("block")
    last, prev = fp["block"].max(), fp["block"].max() - 1
    f_last = fp[fp.block == last].set_index("atom")["FP"]
    f_prev = fp[fp.block == prev].set_index("atom")["FP"]
    rel = (f_last - f_prev).abs() / f_prev.abs().clip(lower=1e-6)
    assert metrics.loc[last, "fp_change"] == pytest.approx(rel.max(), rel=1e-9)
    g = rdf.set_index(["block", "atom", "r_nm"])["g_r"].to_xarray()
    worst = float(nrmsd(g.sel(block=last), g.sel(block=prev)).max())
    assert metrics.loc[last, "rdf_nrmsd"] == pytest.approx(worst, rel=1e-6)


def test_block_data_kept_for_every_block_and_atom(analysis):
    _u, meta, out = analysis
    n_blocks = int(N_FRAMES * DT_PS // (BLOCK_NS * 1000))
    n = meta["n_candidate_atoms"]
    fp = pd.read_csv(out / "fp_values_per_block.csv")
    assert fp["block"].nunique() == n_blocks and len(fp) == n_blocks * n
    contacts = pd.read_csv(out / "contact_occupancy_per_block.csv")
    assert list(contacts.columns) == ["block", "atom", "Contact_occupancy_5A", "Bridge_occupancy", "atom_index",
                                      "Atom_serial", "Atom_name", "Residue_name", "Residue_number", "Chain_ID",
                                      "Min_distance_to_ligand_A", "t_start_ns", "t_end_ns", "n_frames", "t_mid_ns"]
    assert len(contacts) == n_blocks * n
    assert len(pd.read_csv(out / "convergence_metrics.csv")) == n_blocks - 1
    assert len(meta["frames"]["blocks"]) == n_blocks
    assert len(pd.read_csv(out / "rdf_profiles_per_block.csv")) == n_blocks * n * 2001
    ranks = pd.read_csv(out / "fp_ranking_by_block_wide.csv", index_col=0)
    values = pd.read_csv(out / "fp_values_by_block_wide.csv", index_col=0)
    last = values.columns[-1]
    assert ranks[last].idxmin() == values[last].idxmin()  # rank 1 = most negative in every file
    summary = json.loads((out / "convergence_summary.json").read_text())
    assert summary["n_blocks"] == n_blocks and summary["block_ns"] == BLOCK_NS
    assert summary["tolerances"] == {"fp_rel_tol": 0.10, "rdf_nrmsd_tol": 0.15, "spearman_tol": 0.90, "n_stable": 3}


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
    main(["protein", "--complex", str(work / "sys.pdb"), "--topology", str(work / "sys.pdb"),
          "--trajectory", str(work / "traj.xtc"), "--output-dir", str(out), "--block-ns", str(BLOCK_NS),
          "--rmax-nm", "1.5", "--norm-tail-bins", "300"])
    run = load_run(out / "blocks.nc")
    assert (run.attrs["rmax_nm"], run.attrs["norm_tail_bins"], run.attrs["block_ns"]) == (1.5, 300, BLOCK_NS)
    assert run.r_nm.max() < 1.5
    assert {"Residue_name", "Residue_number", "Chain_ID"} <= set(run.coords)
    for name in ("blocks.nc", "candidate_atom_mapping.csv", "fp_values_per_block.csv", "rdf_profiles_per_block.csv",
                 "fp_values_by_block_wide.csv", "fp_ranking_by_block_wide.csv", "convergence_metrics.csv",
                 "convergence_summary.json", "final_per_atom_table.csv",
                 "final_per_atom_table.md",
                 "run_metadata.json"):
        assert (out / name).exists(), name
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    assert set(table["Chain_ID"]) == {"B"}  # the chain written in the complex file
    meta_text = (out / "run_metadata.json").read_text()
    assert str(tmp_path) not in meta_text
    meta = json.loads(meta_text)
    assert meta["candidate_cutoff_A"] == 7.0  # default
    assert set(meta["inputs"]) == {"complex", "topology", "trajectory"} and "versions" in meta
    assert run_metadata(run) == meta


@pytest.mark.parametrize("cutoff", [7.0, 10.0])
def test_explicit_cutoff_selects_exactly_that_pool(tmp_path, cutoff):
    u_complex = _system()
    options = AnalysisOptions(candidate_cutoff_A=cutoff, blocks=BlockOptions(block_ns=BLOCK_NS))
    _u_traj, meta = _run(tmp_path, options, n_frames=4)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv")
    expected = select_candidates(protein_system(u_complex, u_complex), cutoff)
    assert meta["candidate_cutoff_A"] == cutoff
    assert set(table["Atom_index"]) == set(expected.atom_index.values) and len(table) == expected.sizes["atom"]
    assert table["Min_distance_to_ligand_A"].max() <= cutoff


def test_10_A_pool_is_a_superset_of_the_7_A_pool(small_system):
    system = protein_system(small_system, small_system)
    a7 = set(select_candidates(system, 7.0).atom_index.values)
    a10 = set(select_candidates(system, 10.0).atom_index.values)
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
    main(["protein", "--complex", str(work / "sys.pdb"), "--topology", str(work / "sys.pdb"),
          "--trajectory", str(work / "traj.xtc"), "--output-dir", str(out), "--block-ns", str(BLOCK_NS),
          "--candidate-cutoff-A", "10.0"])
    assert json.loads((out / "run_metadata.json").read_text())["candidate_cutoff_A"] == 10.0
    assert pd.read_csv(out / "final_per_atom_table.csv")["Min_distance_to_ligand_A"].max() <= 10.0


def test_missing_input_fails_early(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        app(["protein", "--complex", str(tmp_path / "nope.pdb"), "--topology", "x", "--trajectory", "y",
             "--output-dir", str(tmp_path / "o")], exit_on_error=False, print_error=False)
    assert not (tmp_path / "o").exists()
