"""`oneopes build-cv`: the PLUMED fragment from a g1_g2.yaml, and malformed files rejected."""
import pytest
import yaml
from pydantic import ValidationError

from oneopes.cli import main

G1 = [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}]
G2 = [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}]


def _build_cv(tmp_path, g1_g2_data, *options):
    g1_g2_path = tmp_path / "g1_g2.yaml"
    g1_g2_path.write_text(yaml.safe_dump(g1_g2_data))
    out_path = tmp_path / "plumed_fragment.dat"
    main(["build-cv", "--g1-g2", str(g1_g2_path), "--out", str(out_path), *options])
    return out_path


def test_valid_yaml_produces_same_serial_strings(tmp_path):
    content = _build_cv(tmp_path, {"G1": G1, "G2": G2}).read_text()
    assert "GROUP ATOMS=3,7" in content
    assert "GROUP ATOMS=11,2" in content


def test_without_a_water_group_the_fragment_asks_for_one(tmp_path):
    content = _build_cv(tmp_path, {"G1": G1, "G2": G2}).read_text()
    assert "# TODO: define your own water-oxygen reference group" in content
    assert "GROUPA=G1 GROUPB=WATER_O" in content


def test_a_given_water_group_is_defined_in_the_fragment(tmp_path):
    content = _build_cv(tmp_path, {"G1": G1, "G2": G2}, "--water-group-def", "GROUPA=1305-14858").read_text()
    assert "WATER_O: GROUP GROUPA=1305-14858" in content and "TODO" not in content


@pytest.mark.parametrize("data", [
    {"G2": G2},                                                  # no G1
    {"G1": G1},                                                  # no G2
    {"G1": [{"name": "X1"}, {"name": "X2", "serial": 7}], "G2": G2},  # an atom without serial
], ids=["no G1", "no G2", "no serial"])
def test_malformed_yaml_is_rejected(tmp_path, data):
    with pytest.raises(ValidationError):
        _build_cv(tmp_path, data)
