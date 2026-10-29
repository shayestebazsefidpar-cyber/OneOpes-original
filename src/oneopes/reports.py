"""Tables and files reporting hydration and convergence results.

``blocks`` is a block run (:func:`oneopes.block_run.block_run`); ``hydration`` is one
:func:`oneopes.waterfp.hydration` Dataset.
"""

import json
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Any, TypeVar

import pandas as pd
import xarray as xr

from oneopes.block_run import run_facts, run_summary, save_run
from oneopes.convergence import TRANSITION_VARIABLES
from oneopes.provenance import to_json


FpTable = TypeVar("FpTable", pd.DataFrame, pd.Series)


def fp_ranks(fp: FpTable) -> FpTable:
    """Rank of each atom's FP (per column): 1 = most negative; ties keep the atoms' order."""
    return fp.rank(method="first").astype(int)


def fp_by_block(blocks: xr.Dataset) -> pd.DataFrame:
    """FP with one row per atom and one column per block."""
    return blocks.fp.reset_coords(drop=True).transpose("atom", "block").to_pandas()


def write_block_tables(outdir: Path, blocks: xr.Dataset) -> None:
    """Write the block-wise CSVs to ``outdir``: FP and g(r) per block, FP and FP rank per atom and block,
    and the transitions between blocks (convergence_metrics.csv, from block 1).

    g(r) leaves out the block times, which would repeat on every radius.
    """
    blocks.fp.to_dataframe(name="FP").to_csv(outdir / "fp_values_per_block.csv")
    blocks.g.reset_coords(drop=True).to_dataframe(name="g_r").to_csv(outdir / "rdf_profiles_per_block.csv")
    fp_wide = fp_by_block(blocks)
    fp_wide.to_csv(outdir / "fp_values_by_block_wide.csv")
    fp_ranks(fp_wide).to_csv(outdir / "fp_ranking_by_block_wide.csv")
    transitions = blocks[TRANSITION_VARIABLES].isel(block=slice(1, None))
    transitions.to_dataframe().to_csv(outdir / "convergence_metrics.csv")


def write_summary(path: Path, blocks: xr.Dataset) -> None:
    """Write the run's convergence summary and its facts as one indented JSON object."""
    path.write_text(json.dumps(asdict(run_summary(blocks)) | run_facts(blocks), indent=2))


def write_run_metadata(outdir: Path, metadata: Mapping[str, Any]) -> None:
    """Write run_metadata.json: what a command used (see :func:`oneopes.provenance.provenance`)."""
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "run_metadata.json").write_text(to_json(metadata))


def write_run(outdir: Path, blocks: xr.Dataset, metadata: Mapping[str, Any] | None = None) -> None:
    """Write a block run to ``outdir``: blocks.nc, the block-wise CSVs and convergence_summary.json.

    With ``metadata``, also run_metadata.json, and the same JSON as the run's ``metadata`` attribute.
    """
    if metadata is not None:
        blocks = blocks.assign_attrs(metadata=to_json(metadata))
        write_run_metadata(outdir, metadata)
    save_run(blocks, outdir / "blocks.nc")
    write_block_tables(outdir, blocks)
    write_summary(outdir / "convergence_summary.json", blocks)


def write_hydration_files(hydration: xr.Dataset, outdir: Path) -> None:
    """Write rdf.csv (atom, r_nm, n_r: water-O density, waters/nm^3) and fingerprints.csv (atom, fp)."""
    outdir.mkdir(parents=True, exist_ok=True)
    hydration.density.reset_coords(drop=True).to_dataframe(name="n_r").to_csv(outdir / "rdf.csv")
    hydration.fp.reset_coords(drop=True).to_dataframe(name="fp").to_csv(outdir / "fingerprints.csv")
