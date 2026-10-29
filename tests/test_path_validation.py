"""
Early input validation: a missing required input file is rejected while the
command line is parsed, before anything is written - for build-cv and monitor.
"""
import pytest
from cyclopts import ValidationError

from oneopes.cli import app


def run(tokens):
    """Run ``oneopes`` with errors raised instead of printed and exited."""
    return app(tokens, exit_on_error=False, print_error=False)


def test_build_cv_missing_g1_g2_file_raises_clear_error(tmp_path):
    out_path = tmp_path / "out" / "plumed_fragment.dat"

    with pytest.raises(ValidationError, match="does_not_exist.yaml.*does not exist"):
        run(["build-cv", "--g1-g2", str(tmp_path / "does_not_exist.yaml"), "--out", str(out_path)])

    assert not out_path.exists()


@pytest.mark.parametrize("missing", ["tpr", "xtc"])
def test_monitor_missing_input_raises_before_creating_outdir(tmp_path, monkeypatch, missing):
    present = tmp_path / "present"
    present.write_text("only existence is checked before this point")
    outdir = tmp_path / "outputs" / "convergence"
    files = {"tpr": str(present), "xtc": str(present), missing: f"does_not_exist.{missing}"}
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValidationError, match=f"does_not_exist.{missing}"):
        run(["monitor", "--tpr", files["tpr"], "--xtc", files["xtc"], "--outdir", str(outdir)])

    assert not outdir.exists()
