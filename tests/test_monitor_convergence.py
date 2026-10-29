"""The ligand convergence monitor: `oneopes monitor` end to end on a synthetic ligand in water, and following mdrun."""
import json
import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import asdict
from itertools import islice
from pathlib import Path

import MDAnalysis as mda
import pandas as pd
import psutil
import pytest

from ligand_systems import ligand_in_water
from oneopes.block_run import convergence_summary, facts_of, load_run, metadata_of
from oneopes.cli import main
from oneopes.ligand.g1_g2 import read_g1_g2_yaml, select_g1_g2
from oneopes.ligand import monitor as monitor_module
from oneopes.ligand.monitor import blocks_as_written, mdrun_running, stop_mdrun

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
    assert (outdir / "fp_vs_time.png").exists() and (outdir / "g_per_block.png").exists()

    fingerprints = pd.read_csv(outdir / "selection" / "fingerprints.csv")
    block_2 = pd.read_csv(outdir / "fp_values_per_block.csv").query("block == 2")
    assert list(fingerprints["fp"]) == list(block_2["FP"])
    ligand = mda.Universe(str(tmp_path / "top.pdb")).select_atoms("resname MOL")
    expected = select_g1_g2(ligand, fingerprints["fp"].to_numpy(), fp_round_decimals=3)
    assert read_g1_g2_yaml(outdir / "selection" / "g1_g2.yaml") == expected

    again = tmp_path / "again"
    main(["select-block", "--tpr", str(tmp_path / "top.pdb"), "--run", str(outdir / "blocks.nc"),
          "--outdir", str(again), "--fp-round-decimals", "3"])
    for name in ("g1_g2.yaml", "fingerprints.csv", "rdf.csv"):
        assert (again / name).read_text() == (outdir / "selection" / name).read_text(), name
    metadata = json.loads((again / "run_metadata.json").read_text())
    assert (metadata["block"], metadata["inputs"]["run"]["file"]) == (2, "blocks.nc")
    assert metadata["options"] == {"block": None, "fp_round_decimals": 3}


def test_select_block_picks_any_block_of_a_saved_run(tmp_path):
    outdir = run_monitor(tmp_path, *NEVER_STABLE)
    main(["select-block", "--tpr", str(tmp_path / "top.pdb"), "--run", str(outdir / "blocks.nc"),
          "--outdir", str(tmp_path / "block3"), "--block", "3"])
    fingerprints = pd.read_csv(tmp_path / "block3" / "fingerprints.csv")
    block_3 = pd.read_csv(outdir / "fp_values_per_block.csv").query("block == 3")
    assert list(fingerprints["fp"]) == list(block_3["FP"])
    assert (tmp_path / "block3" / "g1_g2.yaml").exists()


def test_reads_every_block_and_selects_nothing_without_convergence(tmp_path):
    outdir = run_monitor(tmp_path, *NEVER_STABLE)

    summary = json.loads((outdir / "convergence_summary.json").read_text())
    assert (summary["converged"], summary["converged_block"], summary["n_blocks"]) == (False, None, 5)
    assert summary["trajectory_end_ns"] == (N_FRAMES - 1) / 1000
    assert len(pd.read_csv(outdir / "convergence_metrics.csv")) == 4
    run = load_run(outdir / "blocks.nc")
    assert run.sizes["block"] == 5 and facts_of(run)["trajectory_end_ns"] == summary["trajectory_end_ns"]
    assert asdict(convergence_summary(run)) == {name: summary[name] for name in asdict(convergence_summary(run))}
    metadata = json.loads((outdir / "run_metadata.json").read_text())
    assert metadata_of(run) == metadata
    assert {role: entry["file"] for role, entry in metadata["inputs"].items()} == {"tpr": "top.pdb", "xtc": "traj.xtc"}
    assert metadata["options"]["block_options"]["tolerances"]["fp_rel_tol"] == 0
    assert "metadata" not in summary
    assert not (outdir / "selection").exists()


class GrowingTrajectory:
    """A universe whose trajectory, read again with ``load_new``, has the next of ``n_frames`` frames."""

    def __init__(self, *n_frames):
        self.lengths = iter(n_frames)
        self.trajectory = range(0)

    def load_new(self, _path):
        self.trajectory = range(next(self.lengths))


SLEEPER = [sys.executable, "-c", "import time; time.sleep(30)"]


@pytest.fixture
def fake_mdrun():
    """A running process with mdrun on its command line."""
    process = subprocess.Popen([*SLEEPER, "mdrun"])
    yield process.pid
    process.kill()
    process.wait()


def test_while_mdrun_runs_a_block_waits_for_the_frame_after_it(fake_mdrun):
    """The last frame may be half-written, so a block is complete once a later frame exists."""
    u = GrowingTrajectory(2, 3, 4, 5, 7)
    blocks = blocks_as_written(u, Path("prod.xtc"), 2, fake_mdrun, poll_seconds=0)
    assert [(frames, len(u.trajectory)) for frames in islice(blocks, 3)] == [
        (range(2), 3), (range(2, 4), 5), (range(4, 6), 7)]


def test_an_ended_mdrun_is_not_running_even_before_it_is_reaped():
    child = subprocess.Popen([sys.executable, "-c", "pass", "mdrun"])
    try:
        for _ in range(200):
            if psutil.Process(child.pid).status() == psutil.STATUS_ZOMBIE:
                break
            time.sleep(0.05)
        assert not mdrun_running(child.pid)
    finally:
        child.wait()


def test_mdrun_is_recognised_by_its_program_name():
    named = subprocess.Popen(["/opt/gromacs/bin/mdrun_mpi", *SLEEPER[1:]], executable=sys.executable)
    other = subprocess.Popen(SLEEPER)
    try:
        assert mdrun_running(named.pid) and not mdrun_running(other.pid)
    finally:
        for process in (named, other):
            process.kill()
            process.wait()


def test_a_pid_that_is_not_mdrun_stops_the_monitor_before_it_starts(tmp_path, caplog):
    top, traj = ligand_in_water(tmp_path)
    with pytest.raises(SystemExit):
        main(["monitor", str(os.getpid()), "--tpr", top, "--xtc", traj, "--outdir", str(tmp_path / "out")])
    assert not (tmp_path / "out").exists()
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [f"error: pid {os.getpid()} is not a running gmx mdrun; for a finished run, leave the pid out."]


def test_mdrun_is_stopped_before_the_reports_are_written(tmp_path, monkeypatch):
    stopped = []

    def failing_charts(*_):
        raise RuntimeError("charts failed")

    monkeypatch.setattr(monitor_module, "stop_mdrun", stopped.append)
    monkeypatch.setattr(monitor_module, "write_charts", failing_charts)
    with pytest.raises(RuntimeError, match="charts failed"):
        run_monitor(tmp_path, *ALWAYS_STABLE, "--n-stable", "2")
    assert stopped == [None]


def test_only_an_mdrun_process_is_stopped():
    other = subprocess.Popen(SLEEPER)
    mdrun = subprocess.Popen([*SLEEPER, "mdrun"])
    try:
        stop_mdrun(other.pid)
        stop_mdrun(mdrun.pid)
        assert mdrun.wait(timeout=10) == -signal.SIGTERM
        assert other.poll() is None
    finally:
        other.kill()
        mdrun.kill()
        other.wait()


def test_once_mdrun_has_stopped_every_frame_is_complete():
    u = GrowingTrajectory(4, 4, 4)
    assert list(blocks_as_written(u, Path("prod.xtc"), 2, None, poll_seconds=0)) == [range(2), range(2, 4)]
