"""
Early input validation: a missing required input file raises a clear,
early error naming the path - for build-cv and monitor.
"""
import pytest

from oneopes.cli import main


def test_build_cv_missing_g1_g2_file_raises_clear_error(tmp_path, capsys):
    missing_path = tmp_path / "does_not_exist.yaml"
    out_path = tmp_path / "out" / "plumed_fragment.dat"

    with pytest.raises(SystemExit):
        main(["build-cv", "--g1-g2", str(missing_path), "--out", str(out_path)])

    assert str(missing_path) in capsys.readouterr().err
    assert not out_path.exists()


@pytest.mark.parametrize("missing", ["tpr", "xtc"])
def test_monitor_missing_input_raises_before_creating_outdir(tmp_path, monkeypatch, capsys, missing):
    present = tmp_path / "present"
    present.write_text("only existence is checked before this point")
    outdir = tmp_path / "outputs" / "convergence"
    files = {"tpr": str(present), "xtc": str(present), missing: f"does_not_exist.{missing}"}
    monkeypatch.chdir(tmp_path)

    with pytest.raises(SystemExit):
        main(["monitor", "--tpr", files["tpr"], "--xtc", files["xtc"], "--outdir", str(outdir)])

    assert f"does_not_exist.{missing}" in capsys.readouterr().err
    assert not outdir.exists()
