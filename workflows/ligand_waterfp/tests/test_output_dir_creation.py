"""
Regression tests for a real bug caught during manual end-to-end testing
against a real ligand-in-water system: several CLI scripts wrote to `--out`/`--dst`
without first creating the parent directory, crashing with
`OSError: Cannot save file into a non-existent directory`. Each script was
fixed to `os.makedirs(os.path.dirname(...) or ".", exist_ok=True)` before
writing. These tests assert that fix holds by writing into a nested,
not-yet-existing output directory - regressing to the old behavior would
make every test here fail with the original OSError/FileNotFoundError.

g1_g2_selection no longer has its own CLI (PR1: its main() was removed -
run_official_selection.main() now calls write_g1_g2_yaml() directly, in
the same process, right after the vendored selection functions return -
see official_selection/run_official_selection.py and
g1_g2_selection/select_g1_g2.py's module docstrings). The regression
coverage below now calls write_g1_g2_yaml() directly instead of going
through a removed console script.

prepare_ranking_csv.py and run_official_selection.py's own --out path are
NOT covered here - both need a real .tpr with bond connectivity to
exercise meaningfully, which is out of scope for a synthetic unit test.
Both received the same os.makedirs fix (run_official_selection.py's --out
path now goes through write_g1_g2_yaml(), which has this same regression
test below) and were re-verified manually against a real ligand-in-water
system's data - see tests/README.md.
"""
import sys

import pandas as pd
import yaml
from PIL import Image

from ligand_waterfp.g1_g2_selection.select_g1_g2 import write_g1_g2_yaml
from ligand_waterfp.ligand_cv.build_ligand_cv import main as build_ligand_cv_main
from ligand_waterfp.visualization.add_legend import main as add_legend_main
from ligand_waterfp.waterfp.cli import main as waterfp_cli_main


def test_write_g1_g2_yaml_creates_nested_output_dir(tmp_path):
    result = {
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    }
    out_path = tmp_path / "does" / "not" / "exist" / "g1_g2.yaml"
    assert not out_path.parent.exists()

    write_g1_g2_yaml(result, str(out_path))

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


def test_waterfp_fingerprint_subcommand_creates_nested_output_dir(tmp_path, monkeypatch):
    # 600 flat bulk bins for one atom - enough for the default 500-bin tail
    rdf_csv = tmp_path / "rdf.csv"
    pd.DataFrame({
        "atom": "C1",
        "r_nm": [(i + 0.5) * 0.001 for i in range(600)],
        "n_r": 33.4,
    }).to_csv(rdf_csv, index=False)
    out_path = tmp_path / "does" / "not" / "exist" / "fingerprints.csv"
    assert not out_path.parent.exists()

    monkeypatch.setattr(sys, "argv", [
        "ligand-waterfp", "fingerprint",
        "--rdf-csv", str(rdf_csv),
        "--out", str(out_path),
    ])
    waterfp_cli_main()

    assert out_path.exists()
    fp_df = pd.read_csv(out_path)
    assert list(fp_df["atom"]) == ["C1"]
    assert abs(fp_df["fp"][0]) < 1e-8  # flat bulk profile -> FP == 0
