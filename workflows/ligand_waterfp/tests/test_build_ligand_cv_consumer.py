"""
Black-box tests for build_ligand_cv.py's consumption of g1_g2.yaml. PR7:
the loaded YAML is now parsed through SelectionResult.from_dict() before
serials are extracted, so a malformed file is rejected with a clear error
instead of failing deep inside a generator expression. Uses synthetic
atom names/serials - never real scientific results.
"""
import sys

import pytest
import yaml

from ligand_waterfp.ligand_cv.build_ligand_cv import main as build_ligand_cv_main


def _run_build_ligand_cv(monkeypatch, tmp_path, g1_g2_data):
    g1_g2_path = tmp_path / "g1_g2.yaml"
    g1_g2_path.write_text(yaml.safe_dump(g1_g2_data))
    out_path = tmp_path / "plumed_fragment.dat"

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-build-cv",
        "--g1-g2", str(g1_g2_path),
        "--out", str(out_path),
    ])
    build_ligand_cv_main()
    return out_path


def test_valid_yaml_produces_same_serial_strings(tmp_path, monkeypatch):
    out_path = _run_build_ligand_cv(monkeypatch, tmp_path, {
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    })

    content = out_path.read_text()
    assert "GROUP ATOMS=3,7" in content
    assert "GROUP ATOMS=11,2" in content


def test_missing_g1_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        _run_build_ligand_cv(monkeypatch, tmp_path, {
            "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
        })


def test_missing_g2_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        _run_build_ligand_cv(monkeypatch, tmp_path, {
            "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        })


def test_atom_entry_missing_serial_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(KeyError):
        _run_build_ligand_cv(monkeypatch, tmp_path, {
            "G1": [{"name": "X1"}, {"name": "X2", "serial": 7}],
            "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
        })
