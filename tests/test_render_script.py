"""`oneopes render-script`: the PyMOL script that renders the G1/G2 atoms of a g1_g2.yaml."""
from oneopes.cli import main
from oneopes.ligand.g1_g2 import G1G2Selection, write_g1_g2_yaml
from oneopes.render import G1_RGB
from oneopes.render.pymol_script import pymol_color

SELECTION = G1G2Selection.model_validate({
    "G1": [{"name": "C1", "serial": 1}, {"name": "O4", "serial": 4}],
    "G2": [{"name": "C7", "serial": 7}, {"name": "N2", "serial": 9}],
})


def _render_script(tmp_path, *options):
    write_g1_g2_yaml(SELECTION, tmp_path / "g1_g2.yaml")
    (tmp_path / "complex.pdb").write_text("")
    out = tmp_path / "pymol" / "render.py"
    main(["render-script", "--g1-g2", str(tmp_path / "g1_g2.yaml"), "--complex", str(tmp_path / "complex.pdb"),
          "--ligand-resname", "LIG", "--out", str(out), *options])
    return out.read_text()


def test_the_script_selects_the_g1_g2_atoms_of_the_ligand(tmp_path):
    script = _render_script(tmp_path)
    assert 'cmd.select("ligand", "resn LIG")' in script
    assert 'cmd.select("g1", "ligand and name C1+O4")' in script
    assert 'cmd.select("g2", "ligand and name C7+N2")' in script
    assert f'cmd.load({str(tmp_path / "complex.pdb")!r}, "complex")' in script


def test_the_script_is_valid_python(tmp_path):
    compile(_render_script(tmp_path), "render.py", "exec")


def test_g1_is_coloured_as_in_the_legend(tmp_path):
    assert f'cmd.set_color("g1_color", {pymol_color(G1_RGB)})' in _render_script(tmp_path)


def test_pymol_colours_run_from_0_to_1():
    assert pymol_color((255, 0, 51)) == "[1.000, 0.000, 0.200]"


def test_the_image_and_pocket_cutoff_are_options(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    script = _render_script(tmp_path, "--image", "fig/raw.png", "--pocket-cutoff-A", "6.5")
    assert f"cmd.png({str(tmp_path / 'fig' / 'raw.png')!r}, width=2400" in script
    assert "within 6.5 of ligand" in script
