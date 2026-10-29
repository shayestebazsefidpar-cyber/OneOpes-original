"""Commands that write a file (--out, --dst) create its missing parent directories."""
import numpy as np
import yaml
from matplotlib.image import imsave

from oneopes.cli import main


def test_build_ligand_cv_creates_nested_output_dir(tmp_path):
    g1_g2_path = tmp_path / "g1_g2.yaml"
    g1_g2_path.write_text(yaml.safe_dump({
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    }))
    out_path = tmp_path / "does" / "not" / "exist" / "plumed_fragment.dat"
    assert not out_path.parent.exists()

    main(["build-cv", "--g1-g2", str(g1_g2_path), "--out", str(out_path)])

    assert out_path.exists()
    content = out_path.read_text()
    assert "GROUP ATOMS=3,7" in content
    assert "GROUP ATOMS=11,2" in content


def test_add_legend_creates_nested_output_dir(tmp_path):
    src_path = tmp_path / "raw.png"
    imsave(src_path, np.ones((150, 200, 3)))
    dst_path = tmp_path / "does" / "not" / "exist" / "legend.png"
    assert not dst_path.parent.exists()

    main(["add-legend", "--src", str(src_path), "--dst", str(dst_path), "--entry", "Label:detail:255,0,0"])

    assert dst_path.exists()
