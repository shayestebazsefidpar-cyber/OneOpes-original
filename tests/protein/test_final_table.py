"""Final table on synthetic per-atom results."""

import numpy as np
import pandas as pd

from oneopes.protein.final_table import FINAL_COLUMNS, build_final_table, final_table_markdown

REQUIRED = [
    "FP_Rank", "Atom", "Atom_index", "Atom_serial_pdb", "Atom_name", "Residue_name", "Residue_number",
    "Chain_ID", "Min_distance_to_ligand_A", "FP", "Contact_occupancy_5A", "Bridge_occupancy", "SASA_A2",
]


def _inputs():
    """Three candidates, their full-range values and their SASA, all in candidate row order."""
    cand = pd.DataFrame({
        "Atom": ["GLY3_CA_id8", "SER4_OG_id15", "TRP9_NE1_id60"],
        "Atom_index": [7, 14, 59], "Atom_serial_pdb": [8, 15, 60],
        "Atom_name": ["CA", "OG", "NE1"], "Residue_name": ["GLY", "SER", "TRP"],
        "Residue_number": [3, 4, 9], "Chain_ID": ["", "", ""],
        "Min_distance_to_ligand_A": [6.1, 2.9, 8.8],
    })
    values = pd.DataFrame({"FP": [-5.0, -21.0, -0.7], "Contact_occupancy_5A": [0.2, 1.0, 0.0],
                           "Bridge_occupancy": [0.1, 0.4, 0.0]})
    return cand, values, np.array([3.0, 11.5, 25.0])


def test_exact_required_columns_and_no_zone():
    assert FINAL_COLUMNS == REQUIRED
    table = build_final_table(*_inputs())
    assert list(table.columns) == REQUIRED
    assert not any("zone" in c.lower() or "funnel" in c.lower() or "projection" in c.lower() for c in table.columns)


def test_one_row_per_candidate_with_its_own_values():
    t = build_final_table(*_inputs()).set_index("Atom")
    assert len(t) == 3
    assert (t.loc["TRP9_NE1_id60", "FP"], t.loc["TRP9_NE1_id60", "SASA_A2"]) == (-0.7, 25.0)


def test_rank_one_is_most_negative_fp_and_sorted():
    t = build_final_table(*_inputs())
    assert list(t["FP_Rank"]) == [1, 2, 3]
    assert t.loc[0, "Atom"] == "SER4_OG_id15" and t["FP"].is_monotonic_increasing


def test_no_block_or_convergence_columns():
    t = build_final_table(*_inputs())
    assert not any(k in c for c in t.columns for k in ("nRMSD", "Spearman", "change", "stable", "block", "_ns"))


def test_markdown_copy_has_same_rows_and_columns():
    table = build_final_table(*_inputs())
    lines = final_table_markdown(table).splitlines()
    cells = [[cell.strip() for cell in line.strip("|").split("|")] for line in lines]
    assert cells[0] == REQUIRED
    assert len(lines) == 2 + len(table)
    assert [row[1] for row in cells[2:]] == list(table["Atom"])
    assert cells[2][REQUIRED.index("FP")] == "-21.000"
