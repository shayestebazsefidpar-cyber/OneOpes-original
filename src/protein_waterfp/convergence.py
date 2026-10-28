"""FP ranking of the candidate atoms: rank 1 = most negative FP."""

from __future__ import annotations

import pandas as pd


def fp_ranks(fp: dict[str, float], order: list[str]) -> pd.Series:
    """Rank 1 = most negative FP; ties broken by position in ``order``."""
    position = {a: i for i, a in enumerate(order)}
    ranked = sorted(fp, key=lambda a: (fp[a], position[a]))
    rank = {a: i + 1 for i, a in enumerate(ranked)}
    return pd.Series([rank[a] for a in order], index=order, dtype=float)
