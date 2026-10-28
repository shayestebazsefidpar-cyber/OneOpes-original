"""
Early input validation: a missing required input file raises a clear,
early error naming the path - for build_ligand_cv and monitor_convergence.
"""
import sys

import pytest

from ligand_waterfp.ligand_cv.build_ligand_cv import main as build_ligand_cv_main
from ligand_waterfp.convergence.monitor_convergence import main as monitor_main


# --- build_ligand_cv: missing required input fails early and clearly -------

def test_build_ligand_cv_missing_g1_g2_file_raises_clear_error(tmp_path, monkeypatch, capsys):
    missing_path = tmp_path / "does_not_exist.yaml"
    out_path = tmp_path / "out" / "plumed_fragment.dat"

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-build-cv",
        "--g1-g2", str(missing_path),
        "--out", str(out_path),
    ])

    with pytest.raises(SystemExit) as exc_info:
        build_ligand_cv_main()

    assert str(missing_path) in str(exc_info.value) or missing_path.name in str(exc_info.value)
    # fails before writing anything
    assert not out_path.exists()


# --- monitor_convergence: missing --tpr/--xtc fail before touching --------
# --- MDAnalysis or creating the output directory ---------------------------

def test_monitor_missing_tpr_raises_before_creating_outdir(tmp_path, monkeypatch):
    outdir = tmp_path / "outputs" / "convergence"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-monitor",
        "--tpr", "does_not_exist.tpr",
        "--xtc", "does_not_exist.xtc",
        "--outdir", str(outdir),
    ])

    with pytest.raises(SystemExit) as exc_info:
        monitor_main()

    assert "does_not_exist.tpr" in str(exc_info.value)
    assert not outdir.exists()


def test_monitor_missing_xtc_raises_before_creating_outdir(tmp_path, monkeypatch):
    tpr = tmp_path / "prod.tpr"
    tpr.write_text("not a real tpr, only existence is checked before this point")
    outdir = tmp_path / "outputs" / "convergence"

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-monitor",
        "--tpr", str(tpr),
        "--xtc", "does_not_exist.xtc",
        "--outdir", str(outdir),
    ])

    with pytest.raises(SystemExit) as exc_info:
        monitor_main()

    assert "does_not_exist.xtc" in str(exc_info.value)
    assert not outdir.exists()
