"""`oneopes add-legend`: legend entries from g1_g2.yaml or typed by hand, on a raw render."""
import pytest
from PIL import Image

from oneopes.cli import main
from oneopes.errors import InputError
from oneopes.ligand.g1_g2 import SelectionResult, write_g1_g2_yaml
from oneopes.render.add_legend import G1_RGB, G2_RGB, parse_entry, selection_entries

SELECTION = SelectionResult.model_validate({
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


def test_the_legend_from_g1_g2_yaml_is_drawn_on_a_copy_of_the_render(tmp_path):
    Image.new("RGB", (1200, 900), "white").save(tmp_path / "raw.png")
    write_g1_g2_yaml(SELECTION, tmp_path / "g1_g2.yaml")
    main(["add-legend", "--src", str(tmp_path / "raw.png"), "--dst", str(tmp_path / "out.png"),
          "--g1-g2", str(tmp_path / "g1_g2.yaml")])
    drawn = Image.open(tmp_path / "out.png").convert("RGB")
    assert drawn.size == (1200, 900)
    colours = {colour for _, colour in drawn.getcolors(maxcolors=100_000)}
    assert {G1_RGB, G2_RGB} <= colours


def test_a_legend_needs_an_entry(tmp_path):
    Image.new("RGB", (1200, 900), "white").save(tmp_path / "raw.png")
    with pytest.raises(SystemExit):
        main(["add-legend", "--src", str(tmp_path / "raw.png"), "--dst", str(tmp_path / "out.png")])
    assert not (tmp_path / "out.png").exists()
