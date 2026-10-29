"""
Regression tests for a real bug caught during manual end-to-end testing
against a real ligand-in-water system: several CLI scripts wrote to `--out`/`--dst`
without first creating the parent directory, crashing with
`OSError: Cannot save file into a non-existent directory`. Each script was
fixed to `os.makedirs(os.path.dirname(...) or ".", exist_ok=True)` before
writing. These tests assert that fix holds by writing into a nested,
not-yet-existing output directory - regressing to the old behavior would
make every test here fail with the original OSError/FileNotFoundError.
write_g1_g2_yaml() is covered in test_g1_g2_selection.py.
"""
import sys

import yaml
from PIL import Image

from oneopes.ligand.plumed_cv import main as build_ligand_cv_main
from oneopes.render.add_legend import main as add_legend_main


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
