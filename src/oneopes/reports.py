"""Tables and files reporting hydration and convergence results.

``blocks`` is the stacked block measurements of :func:`oneopes.trajectory_blocks.stack_blocks`;
``hydration`` is one :func:`oneopes.waterfp.hydration` Dataset.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, fields
from pathlib import Path

import pandas as pd
import xarray as xr

from oneopes.convergence import ConvergenceSummary, StabilityResult, stable_in_a_row


def fp_ranks(fp: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Rank of each atom's FP (per column): 1 = most negative; ties keep the atoms' order."""
    return fp.rank(method="first").astype(int)


def block_times(blocks: xr.Dataset) -> pd.DataFrame:
    """One row per block: block, t_start_ns, t_end_ns, t_mid_ns."""
    return blocks.reset_coords()[["t_start_ns", "t_end_ns", "t_mid_ns"]].to_dataframe().reset_index()


def fp_per_block(blocks: xr.Dataset) -> pd.DataFrame:
    """One row per (block, atom): block, t_start_ns, t_end_ns, t_mid_ns, atom, FP."""
    table = blocks.fp.to_dataframe(name="FP").reset_index()
    return table[["block", "t_start_ns", "t_end_ns", "t_mid_ns", "atom", "FP"]]


def rdf_per_block(blocks: xr.Dataset) -> pd.DataFrame:
    """One row per (block, atom, r): block, atom, r_nm, g_r."""
    table = blocks.g.to_dataframe(name="g_r").reset_index()
    return table[["block", "atom", "r_nm", "g_r"]]


def fp_by_block(blocks: xr.Dataset) -> pd.DataFrame:
    """FP with one row per atom and one column per block."""
    return blocks.fp.reset_coords(drop=True).transpose("atom", "block").to_pandas()


def transitions_table(stabilities: Sequence[StabilityResult], blocks: xr.Dataset) -> pd.DataFrame:
    """One row per block after the first: its times, its StabilityResult against the block before it,
    and the stable transitions in a row ending there.

    ``stabilities[i]`` compares block i + 1 with block i.
    """
    later_blocks = block_times(blocks).iloc[1:len(stabilities) + 1].reset_index(drop=True)
    stability = pd.DataFrame(map(asdict, stabilities), columns=[field.name for field in fields(StabilityResult)])
    return later_blocks.join(stability).assign(stable_in_a_row=stable_in_a_row(stabilities))


def write_block_tables(outdir: Path, blocks: xr.Dataset, stabilities: Sequence[StabilityResult]) -> None:
    """Write the block-wise CSVs to ``outdir``: FP and g(r) per block, FP and FP rank per atom and block,
    and the transitions between blocks (convergence_metrics.csv)."""
    fp_per_block(blocks).to_csv(outdir / "fp_values_per_block.csv", index=False)
    rdf_per_block(blocks).to_csv(outdir / "rdf_profiles_per_block.csv", index=False)
    fp_wide = fp_by_block(blocks)
    fp_wide.to_csv(outdir / "fp_values_by_block_wide.csv")
    fp_ranks(fp_wide).to_csv(outdir / "fp_ranking_by_block_wide.csv")
    transitions_table(stabilities, blocks).to_csv(outdir / "convergence_metrics.csv", index=False)


def write_summary(path: Path, summary: ConvergenceSummary, **facts) -> None:
    """Write ``summary`` and any other ``facts`` about the run as one indented JSON object."""
    path.write_text(json.dumps({**asdict(summary), **facts}, indent=2))


def write_hydration_files(hydration: xr.Dataset, outdir: Path) -> None:
    """Write rdf.csv (atom, r_nm, n_r: water-O density, waters/nm^3) and fingerprints.csv (atom, fp)."""
    outdir.mkdir(parents=True, exist_ok=True)
    density = hydration.density.to_dataframe(name="n_r").reset_index()
    density[["atom", "r_nm", "n_r"]].to_csv(outdir / "rdf.csv", index=False)
    fp = hydration.fp.to_dataframe(name="fp").reset_index()
    fp[["atom", "fp"]].to_csv(outdir / "fingerprints.csv", index=False)
