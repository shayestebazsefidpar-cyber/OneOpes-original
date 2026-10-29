"""The final per-atom table: FP, Contact and Bridge over the whole analysed trajectory, and static SASA."""

import pandas as pd
import xarray as xr

from oneopes.protein.contacts import CONTACT_CUTOFF_A
from oneopes.reports import fp_ranks

CONTACT_COLUMN = f"Contact_occupancy_{CONTACT_CUTOFF_A:g}A"

TABLE_NAMES = {
    "atom": "Atom",
    "atom_index": "Atom_index",
    "atom_serial": "Atom_serial",
    "atom_name": "Atom_name",
    "residue_name": "Residue_name",
    "residue_number": "Residue_number",
    "chain_id": "Chain_ID",
    "min_distance_to_ligand_A": "Min_distance_to_ligand_A",
    "fp": "FP",
    "contact_occupancy": CONTACT_COLUMN,
    "bridge_occupancy": "Bridge_occupancy",
    "sasa": "SASA_A2",
}
"""The table column of each name in the protein Datasets, in the final table's order."""

FINAL_COLUMNS = ["FP_Rank", *TABLE_NAMES.values()]


def build_final_table(full: xr.Dataset) -> pd.DataFrame:
    """One row per candidate atom, ranked by FP: its coordinates, FP, occupancies and SASA."""
    table = full[["fp", "contact_occupancy", "bridge_occupancy", "sasa"]].to_dataframe().reset_index()
    table = table.rename(columns=TABLE_NAMES)
    table["FP_Rank"] = fp_ranks(table["FP"])
    return table[FINAL_COLUMNS].sort_values("FP_Rank").reset_index(drop=True)


MARKDOWN_DECIMALS = {
    "Min_distance_to_ligand_A": 2,
    "FP": 3,
    CONTACT_COLUMN: 3,
    "Bridge_occupancy": 3,
    "SASA_A2": 2,
}
"""Decimals of the floats in the Markdown copy; the CSV keeps full precision."""


def final_table_markdown(table: pd.DataFrame) -> str:
    """The final table as Markdown: same rows, order and columns; floats rounded."""
    float_formats = [f".{MARKDOWN_DECIMALS[column]}f" if column in MARKDOWN_DECIMALS else "g"
                     for column in map(str, table.columns)]
    return table.to_markdown(index=False, floatfmt=float_formats) + "\n"
