"""`oneopes add-legend`: legend entries from g1_g2.yaml or typed by hand, on a raw render."""
import numpy as np
import pytest
from matplotlib.image import imread, imsave

from oneopes.cli import main
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import G1G2Selection, write_g1_g2_yaml
from oneopes.render import G1_RGB, G2_RGB
from oneopes.render.add_legend import parse_entry, selection_entries

SELECTION = G1G2Selection.model_validate({
    "G1": [{"name": "C1", "serial": 1}, {"name": "O4", "serial": 4}],
    "G2": [{"name": "C7", "serial": 7}, {"name": "N2", "serial": 9}],
})


def test_the_selection_gives_one_red_g1_and_one_blue_g2_entry():
    assert selection_entries(SELECTION) == [("Anti-bulk (G1):  C1, O4", G1_RGB), ("Bulk (G2):  C7, N2", G2_RGB)]


def test_an_entry_typed_by_hand():
    assert parse_entry("Pocket:grey:128,128,128") == ("Pocket:  grey", (128, 128, 128))


@pytest.mark.parametrize("spec", ["Pocket:128,128,128", "Pocket:grey:128,128", "Pocket:grey:red"])
def test_a_malformed_entry_is_an_input_error(spec):
    with pytest.raises(InputError, match="title:detail:r,g,b"):
        parse_entry(spec)


def blank_render(path):
    imsave(path, np.ones((900, 1200, 3)))


def test_the_legend_from_g1_g2_yaml_is_drawn_on_a_copy_of_the_render(tmp_path):
    blank_render(tmp_path / "raw.png")
    write_g1_g2_yaml(SELECTION, tmp_path / "g1_g2.yaml")
    main(["add-legend", "--src", str(tmp_path / "raw.png"), "--dst", str(tmp_path / "out.png"),
          "--g1-g2", str(tmp_path / "g1_g2.yaml")])
    drawn = imread(tmp_path / "out.png")
    assert drawn.shape[:2] == (900, 1200)
    colours = {tuple(pixel) for pixel in np.round(drawn[..., :3] * 255).astype(int).reshape(-1, 3)}
    assert {G1_RGB, G2_RGB} <= colours


def test_a_legend_needs_an_entry(tmp_path):
    blank_render(tmp_path / "raw.png")
    with pytest.raises(SystemExit):
        main(["add-legend", "--src", str(tmp_path / "raw.png"), "--dst", str(tmp_path / "out.png")])
    assert not (tmp_path / "out.png").exists()
