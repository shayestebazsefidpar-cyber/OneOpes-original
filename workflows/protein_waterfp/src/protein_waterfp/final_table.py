"""
The final per-atom table, built only from structured intermediate results.

One row per candidate atom, sorted by FP_Rank (1 = most negative FP). FP,
Contact and Bridge are computed over the full analysed trajectory (not one
block); SASA is static. Block-wise convergence metrics are kept out of this
table (see convergence_metrics.csv); only the analysed time span and the time
of first sustained convergence are carried here.
"""

from __future__ import annotations

import pandas as pd

from .convergence import fp_ranks

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


def build_final_table(candidates: pd.DataFrame, fp: dict, contact: dict, bridge: dict, sasa: dict,
                      analysis_time_ns: float, first_convergence_ns: float | None) -> pd.DataFrame:
    """Merge full-trajectory per-atom results (keyed by ``Atom`` label; SASA by ``Atom_index``).

    ``first_convergence_ns`` is None (written as an empty cell) when sustained
    convergence was never reached.
    """
    labels = list(candidates["Atom"])
    for name, values in (("FP", fp), ("Contact", contact), ("Bridge", bridge)):
        missing = set(labels) - set(values)
        if missing:
            raise ValueError(f"{name} is missing for {len(missing)} candidate atoms.")
    table = candidates.copy()
    table["FP"] = [fp[a] for a in labels]
    table["FP_Rank"] = fp_ranks(fp, labels).astype(int).to_numpy()
    table["Contact_occupancy_5A"] = [contact[a] for a in labels]
    table["Bridge_occupancy"] = [bridge[a] for a in labels]
    table["SASA_A2"] = [sasa[int(i)] for i in table["Atom_index"]]
    table["Analysis_time_ns"] = float(analysis_time_ns)
    table["First_convergence_ns"] = float("nan") if first_convergence_ns is None else float(first_convergence_ns)
    table = table[FINAL_COLUMNS].sort_values("FP_Rank").reset_index(drop=True)
    if table["Atom"].duplicated().any() or len(table) != len(labels):
        raise AssertionError("final table must have exactly one row per candidate atom")
    return table
