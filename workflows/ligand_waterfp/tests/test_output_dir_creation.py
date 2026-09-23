"""
Regression tests for a real bug caught during manual end-to-end testing
against a real ligand-in-water system: several CLI scripts wrote to `--out`/`--dst`
without first creating the parent directory, crashing with
`OSError: Cannot save file into a non-existent directory`. Each script was
fixed to `os.makedirs(os.path.dirname(...) or ".", exist_ok=True)` before
writing. These tests assert that fix holds by writing into a nested,
not-yet-existing output directory - regressing to the old behavior would
make every test here fail with the original OSError/FileNotFoundError.

prepare_ranking_csv.py and run_official_selection.py are NOT covered here
- both need a real .tpr with bond connectivity to exercise meaningfully,
which is out of scope for a synthetic unit test. Both received the same
os.makedirs fix and were re-verified manually against a real
ligand-in-water system's data after the fix - see tests/README.md.
"""
import sys

import yaml
from PIL import Image

from ligand_waterfp.g1_g2_selection.select_g1_g2 import main as select_g1_g2_main
from ligand_waterfp.ligand_cv.build_ligand_cv import main as build_ligand_cv_main
from ligand_waterfp.visualization.add_legend import main as add_legend_main


def test_select_g1_g2_creates_nested_output_dir(tmp_path, monkeypatch):
    selection_output = tmp_path / "selection_output.txt"
    selection_output.write_text(
        "anti-bulk fp selection: X1 (3), X2 (7)\n"
        "bulk fp selection: X3 (11), X4 (2)\n"
    )
    out_path = tmp_path / "does" / "not" / "exist" / "g1_g2.yaml"
    assert not out_path.parent.exists()

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-g1g2",
        "--selection-output", str(selection_output),
        "--out", str(out_path),
    ])
    select_g1_g2_main()

    assert out_path.exists()
    data = yaml.safe_load(out_path.read_text())
    assert data["G1"][0]["name"] == "X1"


def test_build_ligand_cv_creates_nested_output_dir(tmp_path, monkeypatch):
    g1_g2_path = tmp_path / "g1_g2.yaml"
    g1_g2_path.write_text(yaml.safe_dump({
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    }))
    out_path = tmp_path / "does" / "not" / "exist" / "plumed_fragment.dat"
    assert not out_path.parent.exists()

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-build-cv",
        "--g1-g2", str(g1_g2_path),
        "--out", str(out_path),
    ])
    build_ligand_cv_main()

    assert out_path.exists()
    content = out_path.read_text()
    assert "GROUP ATOMS=3,7" in content
    assert "GROUP ATOMS=11,2" in content


def test_add_legend_creates_nested_output_dir(tmp_path, monkeypatch):
    src_path = tmp_path / "raw.png"
    Image.new("RGB", (200, 150), color=(255, 255, 255)).save(src_path)
    dst_path = tmp_path / "does" / "not" / "exist" / "legend.png"
    assert not dst_path.parent.exists()

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp-add-legend",
        "--src", str(src_path),
        "--dst", str(dst_path),
        "--entry", "Label:detail:255,0,0",
    ])
    add_legend_main()

    assert dst_path.exists()
