"""
Unit tests for ligand_waterfp.g1_g2_selection.select_g1_g2's text parser
and (PR3, code-review Rule 3) its AtomRef/SelectionResult dataclasses.
Uses synthetic atom names/serials - never real scientific results.
"""
import dataclasses

import pytest
import yaml

from ligand_waterfp.g1_g2_selection.select_g1_g2 import (
    parse_selection_lines,
    write_g1_g2_yaml,
    AtomRef,
    SelectionResult,
)


def test_parses_both_lines():
    text = (
        "anti-bulk fp selection: X1 (3), X2 (7)\n"
        "bulk fp selection: X3 (11), X4 (2)\n"
    )
    result = parse_selection_lines(text)

    assert result["G1"] == [
        {"name": "X1", "serial": 3},
        {"name": "X2", "serial": 7},
    ]
    assert result["G2"] == [
        {"name": "X3", "serial": 11},
        {"name": "X4", "serial": 2},
    ]


def test_parses_lines_embedded_in_surrounding_log_noise():
    text = (
        "======================================================================\n"
        "Highest-ranking atom: X1 (3) with fp_round -5.0\n"
        "anti-bulk fp selection: X1 (3), X2 (7)\n"
        "======================================================================\n"
        "bulk fp selection: X3 (11), X4 (2)\n"
    )
    result = parse_selection_lines(text)
    assert result["G1"][0]["name"] == "X1"
    assert result["G2"][0]["name"] == "X3"


def test_missing_lines_returns_incomplete_dict():
    result = parse_selection_lines("nothing useful here\n")
    assert "G1" not in result
    assert "G2" not in result


# --- PR3: AtomRef / SelectionResult (code-review Rule 3) ---
# parse_selection_lines() itself is intentionally left returning the
# original dict shape (see select_g1_g2.py's module docstring) - these
# tests cover the new dataclasses directly and the dict<->SelectionResult
# round-trip used at write_g1_g2_yaml()'s serialization boundary.

def test_atom_ref_fields_and_dict_roundtrip():
    a = AtomRef(name="X1", serial=3)
    assert a.name == "X1"
    assert a.serial == 3
    assert a.to_dict() == {"name": "X1", "serial": 3}
    assert AtomRef.from_dict({"name": "X1", "serial": 3}) == a


def test_atom_ref_is_frozen():
    a = AtomRef(name="X1", serial=3)
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.serial = 4


def test_selection_result_from_dict_matches_parse_selection_lines_output():
    text = "anti-bulk fp selection: X1 (3), X2 (7)\nbulk fp selection: X3 (11), X4 (2)\n"
    parsed = parse_selection_lines(text)

    result = SelectionResult.from_dict(parsed)

    assert result.G1 == (AtomRef("X1", 3), AtomRef("X2", 7))
    assert result.G2 == (AtomRef("X3", 11), AtomRef("X4", 2))
    assert result.system_id is None


def test_selection_result_is_frozen():
    result = SelectionResult.from_dict({
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    })
    with pytest.raises(dataclasses.FrozenInstanceError):
        result.system_id = "changed"


def test_selection_result_dict_roundtrip_is_exact():
    """dict -> SelectionResult -> to_yaml_dict() must reproduce the
    original dict exactly - this is the mapping write_g1_g2_yaml() relies
    on to keep the on-disk YAML schema unchanged when given a
    SelectionResult instead of a raw dict."""
    original = {
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
        "system_id": "test-system",
    }

    result = SelectionResult.from_dict(original)
    round_tripped = result.to_yaml_dict()

    assert round_tripped == original


def test_selection_result_yaml_dict_omits_system_id_when_unset():
    result = SelectionResult.from_dict({
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    })
    assert "system_id" not in result.to_yaml_dict()


def test_write_g1_g2_yaml_accepts_selection_result_and_matches_dict_output(tmp_path):
    """write_g1_g2_yaml() must write byte-for-byte the same YAML whether
    given the original dict shape or an equivalent SelectionResult - the
    existing on-disk schema (and existing readers like build_ligand_cv.py)
    must not be able to tell the difference."""
    raw = {
        "G1": [{"name": "X1", "serial": 3}, {"name": "X2", "serial": 7}],
        "G2": [{"name": "X3", "serial": 11}, {"name": "X4", "serial": 2}],
    }
    selection_result = SelectionResult.from_dict(raw)

    dict_out = tmp_path / "from_dict.yaml"
    dataclass_out = tmp_path / "from_dataclass.yaml"
    write_g1_g2_yaml(raw, str(dict_out), system_id="sys-A")
    write_g1_g2_yaml(selection_result, str(dataclass_out), system_id="sys-A")

    assert dict_out.read_text() == dataclass_out.read_text()
    assert yaml.safe_load(dict_out.read_text()) == {**raw, "system_id": "sys-A"}
