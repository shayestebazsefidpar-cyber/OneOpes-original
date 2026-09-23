"""
Unit tests for ligand_waterfp.g1_g2_selection.select_g1_g2's text parser.
Uses synthetic atom names/serials - never real scientific results.
"""
import pytest

from ligand_waterfp.g1_g2_selection.select_g1_g2 import parse_selection_lines


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
