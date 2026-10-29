"""The ligand convergence monitor: its plots, and `oneopes monitor` end to end on a synthetic ligand in water."""
import json
from dataclasses import asdict

import MDAnalysis as mda
import pandas as pd
import xarray as xr

from ligand_systems import ligand_in_water
from oneopes.block_run import load_run, run_facts, run_summary
from oneopes.cli import main
from oneopes.ligand.g1_g2 import read_g1_g2_yaml, select_g1_g2
from oneopes.ligand.monitor import plot_atom

N_FRAMES = 10
BLOCK = ["--block-ns", "0.002"]  # 2 frames of 1 ps: 5 blocks
ALWAYS_STABLE = ["--fp-rel-tol", "100", "--rdf-nrmsd-tol", "100", "--spearman-tol", "-1"]
NEVER_STABLE = ["--fp-rel-tol", "0"]


def run_monitor(tmp_path, *options: str):
    """Run `oneopes monitor` (no mdrun PID) on the synthetic ligand; return the output directory."""
    top, traj = ligand_in_water(tmp_path, n_frames=N_FRAMES)
    outdir = tmp_path / "out"
    main(["monitor", "--tpr", top, "--xtc", traj, "--outdir", str(outdir), *BLOCK, *options])
    return outdir


def test_stops_at_convergence_and_selects_g1_g2_on_the_converged_block(tmp_path):
    """With every transition stable, two in a row converge at block 2; blocks 3 and 4 are never read."""
    outdir = run_monitor(tmp_path, *ALWAYS_STABLE, "--n-stable", "2", "--fp-round-decimals", "3")

    summary = json.loads((outdir / "convergence_summary.json").read_text())
    assert (summary["converged"], summary["converged_block"], summary["n_blocks"]) == (True, 2, 3)
    assert summary["ligand_heavy_atoms"] == [f"C{i}" for i in range(6)]
    assert set(pd.read_csv(outdir / "fp_values_per_block.csv")["block"]) == {0, 1, 2}
    assert len(pd.read_csv(outdir / "convergence_metrics.csv")) == 2
    assert len(list((outdir / "fp_plots").glob("*.png"))) == 6

    fingerprints = pd.read_csv(outdir / "selection" / "fingerprints.csv")
    block_2 = pd.read_csv(outdir / "fp_values_per_block.csv").query("block == 2")
    assert list(fingerprints["fp"]) == list(block_2["FP"])
    ligand = mda.Universe(str(tmp_path / "top.pdb")).select_atoms("resname MOL")
    expected = select_g1_g2(ligand, fingerprints["fp"].to_numpy(), fp_round_decimals=3)
    assert read_g1_g2_yaml(outdir / "selection" / "g1_g2.yaml") == expected


def test_reads_every_block_and_selects_nothing_without_convergence(tmp_path):
    outdir = run_monitor(tmp_path, *NEVER_STABLE)

    summary = json.loads((outdir / "convergence_summary.json").read_text())
    assert (summary["converged"], summary["converged_block"], summary["n_blocks"]) == (False, None, 5)
    assert summary["trajectory_end_ns"] == (N_FRAMES - 1) / 1000
    assert len(pd.read_csv(outdir / "convergence_metrics.csv")) == 4
    run = load_run(outdir / "blocks.nc")
    assert run.sizes["block"] == 5 and run_facts(run)["trajectory_end_ns"] == summary["trajectory_end_ns"]
    assert asdict(run_summary(run)) == {name: summary[name] for name in asdict(run_summary(run))}
    assert not (outdir / "selection").exists()


def test_plot_atom_writes_expected_files(tmp_path):
    (tmp_path / "fp_plots").mkdir()
    (tmp_path / "rdf_plots").mkdir()

    blocks = xr.Dataset(
        {"fp": (("block", "atom"), [[-10.0], [-10.1]]),
         "g": (("block", "atom", "r_nm"), [[[0.0, 1.0]], [[0.0, 1.0]]])},
        coords={"block": [0, 1], "atom": ["A"], "r_nm": [0.1, 0.2], "t_mid_ns": ("block", [2.5, 7.5])},
    )

    fp_plot_path, rdf_plot_path = plot_atom("A", blocks, tmp_path)

    assert fp_plot_path == tmp_path / "fp_plots" / "FP_vs_time_A.png"
    assert rdf_plot_path == tmp_path / "rdf_plots" / "RDF_per_block_A.png"
    assert (tmp_path / "fp_plots" / "FP_vs_time_A.png").exists()
    assert (tmp_path / "rdf_plots" / "RDF_per_block_A.png").exists()
