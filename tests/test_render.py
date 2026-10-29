"""`oneopes render`: the PyMOL script from g1_g2.yaml, run by a stand-in PyMOL, and the legend on its image."""
import logging
import sys
from pathlib import Path

import MDAnalysis as mda
import numpy as np
import pytest
from matplotlib.image import imread

from oneopes.cli import main
from oneopes.ligand.g1_g2 import G1G2Selection, write_g1_g2_yaml
from oneopes.render import G1_RGB, G2_RGB, RENDER_SIZE
from oneopes.render.legend import legend_entries
from oneopes.render.pymol_script import pymol_color

FAKE_PYMOL_PACKAGE = Path(__file__).parent / "fake_pymol"
SELECTION = G1G2Selection.model_validate({
    "G1": [{"name": "C1", "serial": 1}, {"name": "O4", "serial": 4}],
    "G2": [{"name": "C7", "serial": 7}, {"name": "N2", "serial": 9}],
})


def executable(path: Path, body: str) -> str:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(0o755)
    return str(path)


@pytest.fixture
def fake_pymol(tmp_path):
    """A `pymol` that runs the script with the stand-in PyMOL package."""
    return executable(tmp_path / "pymol", f"import runpy, sys\nsys.path.insert(0, {str(FAKE_PYMOL_PACKAGE)!r})\n"
                                          "runpy.run_path(sys.argv[-1])\n")


LIGAND_NAMES = ("C1", "O4", "C7", "N2")


def write_complex(path: Path, residues: list[tuple[str, ...]]) -> None:
    """A complex of residues all named LIG, numbered from 1, with atoms of these names."""
    names = [name for residue in residues for name in residue]
    resindex = [index for index, residue in enumerate(residues) for _ in residue]
    u = mda.Universe.empty(len(names), n_residues=len(residues), atom_resindex=resindex, trajectory=True)
    u.add_TopologyAttr("names", names)
    u.add_TopologyAttr("resnames", ["LIG"] * len(residues))
    u.add_TopologyAttr("resids", list(range(1, len(residues) + 1)))
    u.atoms.positions = np.arange(3 * len(names), dtype=float).reshape(-1, 3)
    u.dimensions = [30.0, 30.0, 30.0, 90.0, 90.0, 90.0]
    u.atoms.write(str(path))


def render(tmp_path, pymol, *options, ligand_selection="resname LIG", residues=(LIGAND_NAMES,)):
    write_g1_g2_yaml(SELECTION, tmp_path / "g1_g2.yaml")
    write_complex(tmp_path / "complex.pdb", list(residues))
    out = tmp_path / "figures" / "selected.png"
    main(["render", "--g1-g2", str(tmp_path / "g1_g2.yaml"), "--complex", str(tmp_path / "complex.pdb"),
          "--ligand-selection", ligand_selection, "--out", str(out), "--pymol", pymol, *options])
    return out


def test_the_figure_the_raw_render_and_the_script_are_written(tmp_path, fake_pymol):
    out = render(tmp_path, fake_pymol)
    assert sorted(path.name for path in out.parent.iterdir()) == [
        "selected.png", "selected_pymol.py", "selected_raw.png"]
    assert imread(out).shape[:2] == RENDER_SIZE[::-1]


def test_the_legend_is_drawn_in_the_render_s_colours(tmp_path, fake_pymol):
    drawn = imread(render(tmp_path, fake_pymol))
    colours = {tuple(pixel) for pixel in np.round(drawn[..., :3] * 255).astype(int).reshape(-1, 3)}
    assert {G1_RGB, G2_RGB} <= colours


def test_the_script_selects_the_g1_g2_atoms_in_the_legend_s_colours(tmp_path, fake_pymol):
    script = (render(tmp_path, fake_pymol).parent / "selected_pymol.py").read_text()
    assert 'cmd.select("ligand", "id 1+2+3+4")' in script
    assert 'cmd.select("g1", "ligand and name C1+O4")' in script
    assert 'cmd.select("g2", "ligand and name C7+N2")' in script
    assert f'cmd.set_color("g1_color", {pymol_color(G1_RGB)})' in script
    assert f'cmd.load({str(tmp_path / "complex.pdb")!r}, "complex")' in script


def test_the_selection_gives_one_red_g1_and_one_blue_g2_entry():
    assert legend_entries(SELECTION) == [("Anti-bulk (G1):  C1, O4", G1_RGB), ("Bulk (G2):  C7, N2", G2_RGB)]


def test_pymol_colours_run_from_0_to_1():
    assert pymol_color((255, 0, 51)) == "[1.000, 0.000, 0.200]"


def error_of(caplog) -> str:
    """The one-line error `oneopes` reported."""
    [error] = [record.getMessage() for record in caplog.records if record.levelno == logging.ERROR]
    return error


def test_only_the_selected_residue_is_the_ligand_when_another_shares_its_name(tmp_path, fake_pymol):
    """As in a complex whose cofactor has the ligand's residue name."""
    out = render(tmp_path, fake_pymol, ligand_selection="resname LIG and resid 2",
                 residues=(("C8", "N9"), LIGAND_NAMES))
    assert 'cmd.select("ligand", "id 3+4+5+6")' in (out.parent / "selected_pymol.py").read_text()


def test_a_selection_that_matches_nothing_is_reported_before_pymol_runs(tmp_path, fake_pymol, caplog):
    with pytest.raises(SystemExit):
        render(tmp_path, fake_pymol, ligand_selection="resname MOL")
    assert error_of(caplog) == "error: --ligand-selection 'resname MOL' matches no atom of complex.pdb."
    assert not (tmp_path / "figures").exists()


def test_an_invalid_selection_is_one_line(tmp_path, fake_pymol, caplog):
    with pytest.raises(SystemExit):
        render(tmp_path, fake_pymol, ligand_selection="resname LIG and")
    assert error_of(caplog).startswith("error: --ligand-selection 'resname LIG and':")


def test_g1_g2_atoms_missing_from_the_ligand_are_named(tmp_path, fake_pymol, caplog):
    with pytest.raises(SystemExit):
        render(tmp_path, fake_pymol, residues=(("C1", "O4", "C7"),))
    assert error_of(caplog) == "error: The ligand (resname LIG) in complex.pdb has no atoms named ['N2']."


def test_a_missing_pymol_is_reported(tmp_path, caplog):
    with pytest.raises(SystemExit):
        render(tmp_path, str(tmp_path / "no_pymol"))
    assert "PyMOL not found" in error_of(caplog)


def test_a_pymol_that_writes_no_image_is_reported_with_its_output(tmp_path, caplog):
    failing = executable(tmp_path / "pymol", "print('Selector-Error: invalid selection')\n")
    with pytest.raises(SystemExit):
        render(tmp_path, failing)
    assert "did not write selected_raw.png" in error_of(caplog) and "Selector-Error" in error_of(caplog)
