"""Final table on synthetic per-atom results."""

import pandas as pd
import pytest

from protein_waterfp.final_table import FINAL_COLUMNS, build_final_table

REQUIRED = [
    "FP_Rank", "Atom", "Atom_index", "Atom_serial_pdb", "Atom_name", "Residue_name", "Residue_number",
    "Chain_ID", "Min_distance_to_ligand_A", "FP", "Contact_occupancy_5A", "Bridge_occupancy", "SASA_A2",
    "Analysis_time_ns", "First_convergence_ns",
]


def _inputs():
    cand = pd.DataFrame({
        "Atom": ["GLY3_CA_id8", "SER4_OG_id15", "TRP9_NE1_id60"],
        "Atom_index": [7, 14, 59], "Atom_serial_pdb": [8, 15, 60],
        "Atom_name": ["CA", "OG", "NE1"], "Residue_name": ["GLY", "SER", "TRP"],
        "Residue_number": [3, 4, 9], "Chain_ID": ["", "", ""],
        "Min_distance_to_ligand_A": [6.1, 2.9, 8.8],
    })
    labels = list(cand["Atom"])
    fp = dict(zip(labels, [-5.0, -21.0, -0.7]))
    contact = dict(zip(labels, [0.2, 1.0, 0.0]))
    bridge = dict(zip(labels, [0.1, 0.4, 0.0]))
    sasa = {7: 3.0, 14: 11.5, 59: 25.0}
    return cand, fp, contact, bridge, sasa, 137.5, 62.5


def test_exact_required_columns_and_no_zone():
    assert FINAL_COLUMNS == REQUIRED
    table = build_final_table(*_inputs())
    assert list(table.columns) == REQUIRED
    assert not any("zone" in c.lower() or "funnel" in c.lower() or "projection" in c.lower() for c in table.columns)


def test_one_row_per_candidate():
    t = build_final_table(*_inputs())
    assert len(t) == 3 and t["Atom"].is_unique


def test_rank_one_is_most_negative_fp_and_sorted():
    t = build_final_table(*_inputs())
    assert list(t["FP_Rank"]) == [1, 2, 3]
    assert t.loc[0, "Atom"] == "SER4_OG_id15" and t["FP"].is_monotonic_increasing


def test_only_minimal_convergence_metadata_and_no_block_metrics():
    t = build_final_table(*_inputs())
    assert set(t["Analysis_time_ns"]) == {137.5} and set(t["First_convergence_ns"]) == {62.5}
    assert not any(k in c for c in t.columns for k in ("nRMSD", "Spearman", "relative_change", "stable", "block"))


def test_not_converged_leaves_first_convergence_empty():
    *args, _first = _inputs()
    t = build_final_table(*args, None)
    assert t["First_convergence_ns"].isna().all()


def test_missing_value_for_a_candidate_fails():
    cand, fp, *rest = _inputs()
    fp.pop("GLY3_CA_id8")
    with pytest.raises(ValueError, match="FP"):
        build_final_table(cand, fp, *rest)


def test_markdown_copy_has_same_rows_and_columns():
    from protein_waterfp.final_table import final_table_markdown

    table = build_final_table(*_inputs())
    lines = final_table_markdown(table).splitlines()
    assert lines[0] == "| " + " | ".join(REQUIRED) + " |"
    assert len(lines) == 2 + len(table)
    rows = [line.strip("| ").split(" | ") for line in lines[2:]]
    assert [r[1] for r in rows] == list(table["Atom"])
    assert rows[0][REQUIRED.index("FP")] == "-21.000"
