"""End-to-end runs on synthetic systems (no funnel, no Zone, no PLUMED)."""

import json

import numpy as np
import pandas as pd
import pytest

from cyclopts import ValidationError

from oneopes.block_run import BlockOptions, estimate_at, load_run, metadata_of
from oneopes.cli import app, main
from oneopes.convergence import CHECK_VARIABLES, ConvergenceCriteria
from oneopes.protein import analysis as analysis_module
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


def _run(tmp_path, options=AnalysisOptions(block_options=BlockOptions(block_ns=BLOCK_NS)), **traj_kw):
    """Run the analysis on the synthetic system; return the trajectory universe and the run metadata."""
    u_traj = _trajectory(_system(), **traj_kw)
    analysis = run_analysis(protein_system(u_traj, _system()), options)
    write_outputs(tmp_path, analysis)
    return u_traj, analysis_details(analysis)


def test_the_charts_come_last_and_show_the_most_negative_fp_candidates(tmp_path, monkeypatch):
    charted = []

    def record(run, outdir):
        charted.append((list(run.atom.values), (outdir / "final_per_atom_table.csv").exists()))

    monkeypatch.setattr(analysis_module, "write_charts", record)
    monkeypatch.setattr(analysis_module, "CHARTED_ATOMS", 3)
    _run(tmp_path)
    [(atoms, table_written_first)] = charted
    assert table_written_first
    assert atoms == list(pd.read_csv(tmp_path / "final_per_atom_table.csv")["Atom"][:3])


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


def test_final_values_are_over_the_production_blocks_the_last_check_found(tmp_path):
    from oneopes.protein.contacts import Contacts
    from oneopes.waterfp import measure_hydration

    n_frames = 13  # 6 blocks of 2 frames + 1 trailing frame, in no block
    u, _meta = _run(tmp_path, n_frames=n_frames)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv").set_index("Atom")
    run = load_run(tmp_path / "blocks.nc")
    production_frames = range(2 * int(run.production_start_block[-1]), 12)
    sel = select_system(u)
    labels = list(pd.read_csv(tmp_path / "candidate_atom_mapping.csv")["Atom"])
    idx = table.loc[labels, "Atom_index"].to_numpy()
    fp = measure_hydration(u.atoms[idx], sel.water(u), production_frames).fp
    contacts = Contacts(u.atoms[idx], sel.ligand(u), sel.water(u)).run(production_frames.start, 12).results
    np.testing.assert_allclose(table.loc[labels, "FP_uncorrected"], fp, rtol=1e-12)
    np.testing.assert_allclose(table.loc[labels, "FP"], estimate_at(run, 5).fp.sel(atom=labels))
    np.testing.assert_allclose(table.loc[labels, "Contact_occupancy_5A"], contacts.contact_occupancy)
    np.testing.assert_allclose(table.loc[labels, "Bridge_occupancy"], contacts.bridge_occupancy)
    assert (table["FP_SE"] > 0).all()


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
    assert meta["frames"].analysed.stop - meta["frames"].blocks[-1].frames.stop == n_frames - n_blocks * fpb
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert len(metrics) == n_blocks
    expected_ends = [t0_ps / 1000.0 + BLOCK_NS * (b + 1) for b in range(n_blocks)]
    np.testing.assert_allclose(metrics["t_end_ns"], expected_ends)
    np.testing.assert_allclose(metrics["t_start_ns"], np.array(expected_ends) - BLOCK_NS)


def test_convergence_reported_retrospectively_and_analysis_continues(tmp_path):
    """A static trajectory: from four production blocks every check passes, so the run converges at block 4,
    the second passing check; all later blocks are still analysed."""
    n_frames = 20  # 10 blocks
    options = AnalysisOptions(block_options=BlockOptions(block_ns=BLOCK_NS, criteria=ConvergenceCriteria(min_blocks=4)))
    _run(tmp_path, options, n_frames=n_frames, noise=0.0)
    summary = json.loads((tmp_path / "convergence_summary.json").read_text())
    metrics = pd.read_csv(tmp_path / "convergence_metrics.csv")
    assert list(metrics["passed"]) == [False] * 3 + [True] * 7
    assert summary["converged"] is True
    assert summary["converged_ns"] == pytest.approx(5 * BLOCK_NS)
    assert summary["converged_block"] == 4
    assert summary["n_blocks"] == 10 and len(metrics) == 10  # nothing truncated
    assert summary["failed_after_convergence"] == 0
    assert pd.read_csv(tmp_path / "fp_values_per_block.csv")["block"].max() == 9
    assert list(metrics["passes_in_a_row"]) == [0, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    assert summary["analysis_end_ns"] - summary["analysis_start_ns"] == pytest.approx((n_frames - 1) * DT_PS / 1000.0)


def test_convergence_metrics_are_a_separate_table(analysis):
    _u, _meta, out = analysis
    metrics = pd.read_csv(out / "convergence_metrics.csv")
    for col in ("block", "t_start_ns", "t_end_ns", *CHECK_VARIABLES, "passes_in_a_row"):
        assert col in metrics.columns, col
    table = pd.read_csv(out / "final_per_atom_table.csv")
    assert not set(metrics.columns) & set(table.columns)


def test_the_last_check_scores_the_final_values(analysis):
    _u, _meta, out = analysis
    last = pd.read_csv(out / "convergence_metrics.csv").iloc[-1]
    table = pd.read_csv(out / "final_per_atom_table.csv")
    assert last["max_drift_z"] == pytest.approx(estimate_at(load_run(out / "blocks.nc"), 5).drift_z.max())
    assert last["goal_score"] == pytest.approx((table["FP_SE"] / table["FP"].abs()).max())


def test_block_data_kept_for_every_block_and_atom(analysis):
    _u, meta, out = analysis
    n_blocks = int(N_FRAMES * DT_PS // (BLOCK_NS * 1000))
    n = meta["n_candidate_atoms"]
    fp = pd.read_csv(out / "fp_values_per_block.csv")
    assert fp["block"].nunique() == n_blocks and len(fp) == n_blocks * n
    contacts = pd.read_csv(out / "contact_occupancy_per_block.csv")
    assert list(contacts.columns) == ["block", "Atom", "Contact_occupancy_5A", "Bridge_occupancy", "Atom_index",
                                      "Atom_serial", "Atom_name", "Residue_name", "Residue_number", "Chain_ID",
                                      "Min_distance_to_ligand_A", "t_start_ns", "t_end_ns", "n_frames", "t_mid_ns"]
    assert len(contacts) == n_blocks * n
    assert len(pd.read_csv(out / "convergence_metrics.csv")) == n_blocks
    assert len(meta["frames"].blocks) == n_blocks
    assert len(pd.read_csv(out / "rdf_profiles_per_block.csv")) == n_blocks * n * 2001
    ranks = pd.read_csv(out / "fp_ranking_by_block_wide.csv", index_col=0)
    values = pd.read_csv(out / "fp_values_by_block_wide.csv", index_col=0)
    last = values.columns[-1]
    assert ranks[last].idxmin() == values[last].idxmin()  # rank 1 = most negative in every file
    summary = json.loads((out / "convergence_summary.json").read_text())
    assert summary["n_blocks"] == n_blocks and summary["block_ns"] == BLOCK_NS
    assert summary["criteria"] == {"min_blocks": 10, "drift_alpha": 0.05, "n_passes": 2}


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
    assert {"residue_name", "residue_number", "chain_id"} <= set(run.coords)
    for name in ("blocks.nc", "candidate_atom_mapping.csv", "fp_values_per_block.csv", "rdf_profiles_per_block.csv",
                 "fp_values_by_block_wide.csv", "fp_ranking_by_block_wide.csv", "convergence_metrics.csv",
                 "convergence_summary.json", "final_per_atom_table.csv", "final_per_atom_table.md",
                 "fp_vs_time.png", "g_per_block.png", "run_metadata.json"):
        assert (out / name).exists(), name
    table = pd.read_csv(out / "final_per_atom_table.csv", keep_default_na=False)
    assert set(table["Chain_ID"]) == {"B"}  # the chain written in the complex file
    meta_text = (out / "run_metadata.json").read_text()
    assert str(tmp_path) not in meta_text
    meta = json.loads(meta_text)
    assert meta["options"]["candidate_cutoff_A"] == 7.0  # default
    assert set(meta["inputs"]) == {"complex", "topology", "trajectory"} and "versions" in meta
    assert metadata_of(run) == meta


@pytest.mark.parametrize("cutoff", [7.0, 10.0])
def test_explicit_cutoff_selects_exactly_that_pool(tmp_path, cutoff):
    u_complex = _system()
    options = AnalysisOptions(candidate_cutoff_A=cutoff, block_options=BlockOptions(block_ns=BLOCK_NS))
    _u_traj, meta = _run(tmp_path, options, n_frames=4)
    table = pd.read_csv(tmp_path / "final_per_atom_table.csv")
    expected = select_candidates(protein_system(u_complex, u_complex), cutoff)
    assert meta["options"].candidate_cutoff_A == cutoff
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
    assert json.loads((out / "run_metadata.json").read_text())["options"]["candidate_cutoff_A"] == 10.0
    assert pd.read_csv(out / "final_per_atom_table.csv")["Min_distance_to_ligand_A"].max() <= 10.0


def test_missing_input_fails_early(tmp_path):
    with pytest.raises(ValidationError, match="does not exist"):
        app(["protein", "--complex", str(tmp_path / "nope.pdb"), "--topology", "x", "--trajectory", "y",
             "--output-dir", str(tmp_path / "o")], exit_on_error=False, print_error=False)
    assert not (tmp_path / "o").exists()
