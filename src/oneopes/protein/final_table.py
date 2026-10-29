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
    "Analysis_time_ns",
    "First_convergence_ns",
]


def build_final_table(candidates: pd.DataFrame, fp: np.ndarray, contact: np.ndarray, bridge: np.ndarray,
                      sasa: np.ndarray, analysis_time_ns: float, first_convergence_ns: float | None) -> pd.DataFrame:
    """The final table from full-trajectory per-atom values, each in ``candidates`` row order.

    ``first_convergence_ns`` is None (written as an empty cell) when sustained
    convergence was never reached. Raises ValueError when a value array does not
    have one value per candidate.
    """
    table = candidates.assign(
        FP=fp,
        FP_Rank=fp_ranks(pd.Series(fp)),
        Contact_occupancy_5A=contact,
        Bridge_occupancy=bridge,
        SASA_A2=sasa,
        Analysis_time_ns=float(analysis_time_ns),
        First_convergence_ns=np.nan if first_convergence_ns is None else float(first_convergence_ns),
    )
    return table[FINAL_COLUMNS].sort_values("FP_Rank").reset_index(drop=True)


MARKDOWN_DECIMALS = {
    "Min_distance_to_ligand_A": 2,
    "FP": 3,
    "Contact_occupancy_5A": 3,
    "Bridge_occupancy": 3,
    "SASA_A2": 2,
    "Analysis_time_ns": 2,
    "First_convergence_ns": 2,
}
"""Decimals of the floats in the Markdown copy; the CSV keeps full precision."""


def final_table_markdown(table: pd.DataFrame) -> str:
    """The final table as Markdown: same rows, order and columns; floats rounded, missing values empty."""
    float_formats = [f".{MARKDOWN_DECIMALS[column]}f" if column in MARKDOWN_DECIMALS else "g" for column in table]
    cells = table.astype(object).where(table.notna(), None)
    return cells.to_markdown(index=False, missingval="", floatfmt=float_formats) + "\n"
