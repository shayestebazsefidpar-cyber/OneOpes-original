"""The run notebook executes from start to finish on a real monitor run (needs the notebooks extra)."""
import subprocess
import sys
from pathlib import Path

import pytest

from ligand_systems import ligand_in_water
from oneopes.cli import main

NOTEBOOKS = Path(__file__).resolve().parents[1] / "notebooks"


def test_explore_run_notebook_runs_on_a_monitor_run(tmp_path):
    pytest.importorskip("marimo")
    top, traj = ligand_in_water(tmp_path, n_frames=6)
    main(["monitor", "--tpr", top, "--xtc", traj, "--outdir", str(tmp_path / "out"), "--block-ns", "0.002"])
    notebook = subprocess.run([sys.executable, str(NOTEBOOKS / "explore_run.py"), "--",
                               "--run", str(tmp_path / "out" / "blocks.nc")], capture_output=True, text=True)
    assert notebook.returncode == 0, notebook.stderr
