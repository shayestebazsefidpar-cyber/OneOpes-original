"""
The final per-atom table: one row per candidate atom, sorted by FP_Rank (1 = most negative FP).

FP, Contact and Bridge are computed over the full analysed trajectory (not one block); SASA is
static. Block-wise convergence is reported separately (convergence_metrics.csv), and the analysed
time span and the time of first sustained convergence are in convergence_summary.json.
"""

import numpy as np
import pandas as pd

from oneopes.convergence import fp_ranks

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
]


def build_final_table(candidates: pd.DataFrame, values: pd.DataFrame, sasa: np.ndarray) -> pd.DataFrame:
    """The candidates with their values (FP, Contact_occupancy_5A, Bridge_occupancy) and SASA, ranked by FP.

    ``values`` and ``sasa`` are in ``candidates`` row order.
    """
    table = candidates.join(values).assign(SASA_A2=sasa)
    table["FP_Rank"] = fp_ranks(table["FP"])
    return table[FINAL_COLUMNS].sort_values("FP_Rank").reset_index(drop=True)


MARKDOWN_DECIMALS = {
    "Min_distance_to_ligand_A": 2,
    "FP": 3,
    "Contact_occupancy_5A": 3,
    "Bridge_occupancy": 3,
    "SASA_A2": 2,
}
"""Decimals of the floats in the Markdown copy; the CSV keeps full precision."""


def final_table_markdown(table: pd.DataFrame) -> str:
    """The final table as Markdown: same rows, order and columns; floats rounded."""
    float_formats = [f".{MARKDOWN_DECIMALS[column]}f" if column in MARKDOWN_DECIMALS else "g"
                     for column in map(str, table.columns)]
    return table.to_markdown(index=False, floatfmt=float_formats) + "\n"
