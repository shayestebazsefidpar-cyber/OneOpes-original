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

import MDAnalysis as mda
import pandas as pd
import psutil
import pytest

from ligand_systems import ligand_in_water
from oneopes.block_run import convergence_summary, estimate_at, facts_of, load_run, metadata_of
from oneopes.cli import main
from oneopes.ligand.g1_g2 import read_g1_g2_yaml, select_g1_g2
from oneopes.ligand import monitor as monitor_module
from oneopes.errors import InputError
from oneopes.ligand.monitor import blocks_as_written, mdrun_running, stop_mdrun, wait_for_frames

N_FRAMES = 16
BLOCK = ["--block-ns", "0.002"]  # 2 frames of 1 ps: 8 blocks
PASSING = ["--min-blocks", "4", "--drift-alpha", "0", "--anchor-confidence", "0"]
"""Every check with four production blocks passes: the checks after blocks 3 and 4 converge the run at block 4."""
NEVER_PASSING = ["--min-blocks", "1000"]


def run_monitor(tmp_path, *options: str):
    """Run `oneopes monitor` (no mdrun PID) on the synthetic ligand; return the output directory."""
    top, traj = ligand_in_water(tmp_path, n_frames=N_FRAMES)
    outdir = tmp_path / "out"
    main(["monitor", "--tpr", top, "--xtc", traj, "--outdir", str(outdir), *BLOCK, *options])
    return outdir


def test_stops_at_convergence_and_selects_g1_g2_on_the_production_blocks(tmp_path):
    """Blocks 5 to 7 are never read."""
    outdir = run_monitor(tmp_path, *PASSING, "--fp-round-decimals", "3")

    summary = json.loads((outdir / "convergence_summary.json").read_text())
    assert (summary["converged"], summary["converged_block"], summary["n_blocks"]) == (True, 4, 5)
    assert summary["ligand_heavy_atoms"] == [f"C{i}" for i in range(6)]
    assert set(pd.read_csv(outdir / "fp_values_per_block.csv")["block"]) == {0, 1, 2, 3, 4}
    assert len(pd.read_csv(outdir / "convergence_metrics.csv")) == 5
    assert (outdir / "fp_vs_time.png").exists() and (outdir / "g_per_block.png").exists()

    fingerprints = pd.read_csv(outdir / "selection" / "fingerprints.csv")
    estimate = estimate_at(load_run(outdir / "blocks.nc"), 4)
    assert list(fingerprints["fp"]) == list(estimate.fp.values)
    assert list(fingerprints.columns) == ["atom", "fp", "fp_se", "fp_uncorrected", "drift_z"]
    ligand = mda.Universe(str(tmp_path / "top.pdb")).select_atoms("resname MOL")
    expected = select_g1_g2(ligand, fingerprints["fp"].to_numpy(), fp_round_decimals=3)
    assert read_g1_g2_yaml(outdir / "selection" / "g1_g2.yaml") == expected

    again = tmp_path / "again"
    main(["select-block", "--tpr", str(tmp_path / "top.pdb"), "--run", str(outdir / "blocks.nc"),
          "--outdir", str(again), "--fp-round-decimals", "3"])
    draws = pd.read_csv(outdir / "selection" / "g1_g2_draws.csv")
    assert draws["fraction"].sum() == pytest.approx(1.0)
    for name in ("g1_g2.yaml", "fingerprints.csv", "rdf.csv", "g1_g2_draws.csv"):
        assert (again / name).read_text() == (outdir / "selection" / name).read_text(), name
    metadata = json.loads((again / "run_metadata.json").read_text())
    assert (metadata["block"], metadata["inputs"]["run"]["file"]) == (4, "blocks.nc")
    assert metadata["options"] == {"block": None, "fp_round_decimals": 3}


def test_select_block_selects_on_the_production_any_check_found(tmp_path):
    outdir = run_monitor(tmp_path, *NEVER_PASSING)
    main(["select-block", "--tpr", str(tmp_path / "top.pdb"), "--run", str(outdir / "blocks.nc"),
          "--outdir", str(tmp_path / "block5"), "--block", "5"])
    fingerprints = pd.read_csv(tmp_path / "block5" / "fingerprints.csv")
    assert list(fingerprints["fp"]) == list(estimate_at(load_run(outdir / "blocks.nc"), 5).fp.values)
    assert (tmp_path / "block5" / "g1_g2.yaml").exists()


def test_reads_every_block_and_selects_nothing_without_convergence(tmp_path):
    outdir = run_monitor(tmp_path, *NEVER_PASSING)

    summary = json.loads((outdir / "convergence_summary.json").read_text())
    assert (summary["converged"], summary["converged_block"], summary["n_blocks"]) == (False, None, 8)
    assert summary["trajectory_end_ns"] == (N_FRAMES - 1) / 1000
    assert len(pd.read_csv(outdir / "convergence_metrics.csv")) == 8
    run = load_run(outdir / "blocks.nc")
    assert run.sizes["block"] == 8 and facts_of(run)["trajectory_end_ns"] == summary["trajectory_end_ns"]
    assert asdict(convergence_summary(run)) == {name: summary[name] for name in asdict(convergence_summary(run))}
    metadata = json.loads((outdir / "run_metadata.json").read_text())
    assert metadata_of(run) == metadata
    assert {role: entry["file"] for role, entry in metadata["inputs"].items()} == {"tpr": "top.pdb", "xtc": "traj.xtc"}
    assert metadata["options"]["block_options"]["criteria"]["min_blocks"] == 1000
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


def test_while_mdrun_runs_a_block_waits_for_the_frame_after_it(tmp_path, fake_mdrun):
    """The last frame may be half-written, so a block is complete once a later frame exists."""
    (tmp_path / "prod.xtc").touch()
    u = GrowingTrajectory(2, 3, 4, 5, 7)
    blocks = blocks_as_written(u, tmp_path / "prod.xtc", 2, fake_mdrun, poll_seconds=0)
    assert [(frames, len(u.trajectory)) for frames in islice(blocks, 3)] == [
        (range(2), 3), (range(2, 4), 5), (range(4, 6), 7)]


def test_a_trajectory_mdrun_has_not_created_yet_is_waited_for(tmp_path, fake_mdrun, monkeypatch):
    """Just after mdrun starts, the trajectory may not exist; here it appears at the first poll."""
    xtc = tmp_path / "prod.xtc"
    monkeypatch.setattr(monitor_module.time, "sleep", lambda _seconds: xtc.touch())
    u = GrowingTrajectory(3)
    wait_for_frames(u, xtc, 2, fake_mdrun, poll_seconds=0)
    assert len(u.trajectory) == 3


def test_a_trajectory_that_no_mdrun_will_write_is_an_error(tmp_path):
    with pytest.raises(InputError, match="does not exist, and no mdrun is running"):
        wait_for_frames(GrowingTrajectory(), tmp_path / "prod.xtc", 2, None, poll_seconds=0)


def test_without_a_pid_the_trajectory_must_exist(tmp_path, caplog):
    top, _ = ligand_in_water(tmp_path)
    with pytest.raises(SystemExit):
        main(["monitor", "--tpr", top, "--xtc", str(tmp_path / "missing.xtc"), "--outdir", str(tmp_path / "out")])
    errors = [r.getMessage() for r in caplog.records if r.levelno == logging.ERROR]
    assert errors == [f"error: {tmp_path / 'missing.xtc'} does not exist; to follow an mdrun that will write it, "
                      "give its pid."]


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
        run_monitor(tmp_path, *PASSING)
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


def test_once_mdrun_has_stopped_every_frame_is_complete(tmp_path):
    (tmp_path / "prod.xtc").touch()
    u = GrowingTrajectory(4, 4, 4)
    assert list(blocks_as_written(u, tmp_path / "prod.xtc", 2, None, poll_seconds=0)) == [range(2), range(2, 4)]
