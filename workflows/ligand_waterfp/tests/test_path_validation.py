"""
Tests for path handling and early input validation: valid str/Path inputs
produce the expected output locations, and a missing required input file
raises a clear, early error naming the path - for build_ligand_cv,
run_official_selection, and monitor_convergence.
"""
import sys

import pandas as pd
import pytest
import yaml

from ligand_waterfp.g1_g2_selection.select_g1_g2 import write_g1_g2_yaml
from ligand_waterfp.ligand_cv.build_ligand_cv import main as build_ligand_cv_main
from ligand_waterfp.official_selection.run_official_selection import main as run_official_selection_main
from ligand_waterfp.convergence.monitor_convergence import main as monitor_main


# --- pathlib inputs behave the same as string inputs -----------------------

def test_write_g1_g2_yaml_accepts_a_path_object(tmp_path):
    """out_path may be a pathlib.Path (not just a str) and lands at the
    same location a string path would - the pathlib migration does not
    change where files end up."""
    result = {
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    }
    out_path = tmp_path / "nested" / "g1_g2.yaml"

    write_g1_g2_yaml(result, out_path)  # Path, not str(out_path)

    assert out_path.exists()
    assert yaml.safe_load(out_path.read_text())["G1"][0]["name"] == "X1"


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


# --- run_official_selection: missing inputs fail early, before any of the --
# --- vendored algorithm's own output is printed -----------------------------

def test_run_official_selection_missing_ranking_csv_raises_clear_error(tmp_path, monkeypatch, capsys):
    missing_csv = tmp_path / "ranking.csv"

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-select",
        "--ranking-csv", str(missing_csv),
        "--system-id", "SystemA-lig1",
    ])

    with pytest.raises(SystemExit) as exc_info:
        run_official_selection_main()

    assert str(missing_csv) in str(exc_info.value) or missing_csv.name in str(exc_info.value)
    # never reached the ranking-table printout
    assert "Ranking table" not in capsys.readouterr().out


def test_run_official_selection_missing_derived_tpr_raises_clear_error(tmp_path, monkeypatch, capsys):
    ranking_csv = tmp_path / "ranking.csv"
    pd.DataFrame({
        "name": ["SystemA-lig1", "SystemA-lig1"],
        "atom": [1, 2],
        "fp": [-10.0, -5.0],
    }).to_csv(ranking_csv, index=False)

    monkeypatch.chdir(tmp_path)  # so the derived "SystemAlig1.tpr" is looked up here
    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-select",
        "--ranking-csv", str(ranking_csv),
        "--system-id", "SystemA-lig1",
    ])

    with pytest.raises(SystemExit) as exc_info:
        run_official_selection_main()

    assert "SystemAlig1.tpr" in str(exc_info.value)
    assert "Ranking table" not in capsys.readouterr().out


def test_run_official_selection_malformed_system_id_raises_clear_error(tmp_path, monkeypatch):
    ranking_csv = tmp_path / "ranking.csv"
    pd.DataFrame({"name": ["nohyphen"], "atom": [1], "fp": [-10.0]}).to_csv(ranking_csv, index=False)

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-select",
        "--ranking-csv", str(ranking_csv),
        "--system-id", "nohyphen",
    ])

    with pytest.raises(SystemExit) as exc_info:
        run_official_selection_main()

    assert "must contain a '-'" in str(exc_info.value)


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
