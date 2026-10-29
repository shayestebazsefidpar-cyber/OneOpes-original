"""
`oneopes build-cv` reading g1_g2.yaml: valid files give the expected serials,
malformed ones are rejected with a ValidationError naming the problem.
Synthetic atom names and serials only.
"""
import pytest
import yaml
from pydantic import ValidationError

from oneopes.cli import main

G1 = [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}]
G2 = [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}]


def _run_build_ligand_cv(tmp_path, g1_g2_data):
    g1_g2_path = tmp_path / "g1_g2.yaml"
    g1_g2_path.write_text(yaml.safe_dump(g1_g2_data))
    out_path = tmp_path / "plumed_fragment.dat"
    main(["build-cv", "--g1-g2", str(g1_g2_path), "--out", str(out_path)])
    return out_path


def test_valid_yaml_produces_same_serial_strings(tmp_path):
    content = _run_build_ligand_cv(tmp_path, {"G1": G1, "G2": G2}).read_text()
    assert "GROUP ATOMS=3,7" in content
    assert "GROUP ATOMS=11,2" in content


@pytest.mark.parametrize("data", [
    {"G2": G2},                                                  # no G1
    {"G1": G1},                                                  # no G2
    {"G1": [{"name": "X1"}, {"name": "X2", "serial": 7}], "G2": G2},  # an atom without serial
], ids=["no G1", "no G2", "no serial"])
def test_malformed_yaml_is_rejected(tmp_path, data):
    with pytest.raises(ValidationError):
        _run_build_ligand_cv(tmp_path, data)
