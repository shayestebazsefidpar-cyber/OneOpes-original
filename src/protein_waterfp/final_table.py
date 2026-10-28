"""
The final per-atom table, built only from structured intermediate results.

One row per candidate atom, sorted by FP_Rank (1 = most negative FP). FP,
Contact and Bridge are computed over the full analysed trajectory (not one
block); SASA is static. Block-wise convergence metrics are kept out of this
table (see convergence_metrics.csv); only the analysed time span and the time
of first sustained convergence are carried here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from convergence import fp_ranks

FINAL_COLUMNS = [
    "FP_Rank",
    "Atom",
    "Atom_index",
    "Atom_serial_pdb",
    "Atom_name",
    "Residue_name",
    "Residue_number",
    "Chain_ID",
    "Min_distance_to_ligand_A",
    "FP",
    "Contact_occupancy_5A",
    "Bridge_occupancy",
    "SASA_A2",
    "Analysis_time_ns",
    "First_convergence_ns",
]


def build_final_table(candidates: pd.DataFrame, fp: np.ndarray, contact: np.ndarray, bridge: np.ndarray,
                      sasa: np.ndarray, analysis_time_ns: float, first_convergence_ns: float | None) -> pd.DataFrame:
    """The final table from full-trajectory per-atom values, each in ``candidates`` row order.

    ``first_convergence_ns`` is None (written as an empty cell) when sustained
    convergence was never reached.
    """
    labels = list(candidates["Atom"])
    for name, values in (("FP", fp), ("Contact", contact), ("Bridge", bridge), ("SASA", sasa)):
        if len(values) != len(labels):
            raise ValueError(f"{name} has {len(values)} values for {len(labels)} candidate atoms.")
    table = candidates.copy()
    table["FP"] = fp
    table["FP_Rank"] = fp_ranks(table["FP"])
    table["Contact_occupancy_5A"] = contact
    table["Bridge_occupancy"] = bridge
    table["SASA_A2"] = sasa
    table["Analysis_time_ns"] = float(analysis_time_ns)
    table["First_convergence_ns"] = float("nan") if first_convergence_ns is None else float(first_convergence_ns)
    table = table[FINAL_COLUMNS].sort_values("FP_Rank").reset_index(drop=True)
    if table["Atom"].duplicated().any() or len(table) != len(labels):
        raise AssertionError("final table must have exactly one row per candidate atom")
    return table


# Decimals used only for the human-readable copy; the CSV keeps full precision.
MARKDOWN_DECIMALS = {
    "Min_distance_to_ligand_A": 2,
    "FP": 3,
    "Contact_occupancy_5A": 3,
    "Bridge_occupancy": 3,
    "SASA_A2": 2,
    "Analysis_time_ns": 2,
    "First_convergence_ns": 2,
}


def final_table_markdown(table: pd.DataFrame) -> str:
    """The final table as a Markdown table (same rows, order and columns; floats rounded)."""

    def cell(column, value):
        if pd.isna(value):
            return ""
        if column in MARKDOWN_DECIMALS:
            return f"{float(value):.{MARKDOWN_DECIMALS[column]}f}"
        return str(value)

    columns = list(table.columns)
    lines = ["| " + " | ".join(columns) + " |", "|" + "|".join("---" for _ in columns) + "|"]
    for row in table.itertuples(index=False):
        lines.append("| " + " | ".join(cell(c, v) for c, v in zip(columns, row)) + " |")
    return "\n".join(lines) + "\n"
