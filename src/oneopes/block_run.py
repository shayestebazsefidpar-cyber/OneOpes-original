"""A block run: the measured trajectory blocks, each compared with the one before, as one Dataset (blocks.nc).

Its attributes are the settings it was made with (the :class:`BlockOptions` and
:class:`~oneopes.waterfp.HydrationOptions`) and any facts about the run, e.g. the ligand selection;
the convergence summary is derived from it, not stored.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated

import numpy as np
import xarray as xr
from cyclopts import Parameter
from cyclopts.validators import Number

from oneopes.convergence import DEFAULT_TOLERANCES, ConvergenceSummary, Tolerances, compare_blocks, summarize
from oneopes.errors import InputError
from oneopes.trajectory_blocks import BLOCK_NS_DEFAULT, stack_blocks
from oneopes.waterfp import HydrationOptions


@dataclass(frozen=True)
class BlockOptions:
    """How long a block is, and when consecutive blocks count as converged; also command options."""
    block_ns: Annotated[float, Parameter(validator=Number(gt=0))] = BLOCK_NS_DEFAULT
    """Trajectory block size (ns)."""
    tolerances: Annotated[Tolerances, Parameter(name="*")] = DEFAULT_TOLERANCES


def settings(blocks: BlockOptions, hydration: HydrationOptions) -> dict[str, float]:
    """The options a run is made with, as flat attributes."""
    return {"block_ns": blocks.block_ns, **asdict(blocks.tolerances), **asdict(hydration.grid),
            "norm_tail_bins": hydration.norm_tail_bins}


SETTINGS = list(settings(BlockOptions(), HydrationOptions()))
"""The attributes holding the settings of a run."""


def block_run(measured: Sequence[xr.Dataset], blocks: BlockOptions, hydration: HydrationOptions,
              **facts) -> xr.Dataset:
    """The measured blocks, stacked and compared (:func:`compare_blocks`), with the options they were made
    with and ``facts`` (strings, numbers or lists of them) as attributes."""
    compared = compare_blocks(stack_blocks(measured), blocks.tolerances)
    return compared.assign_attrs(**settings(blocks, hydration), **facts)


def run_block_options(run: xr.Dataset) -> BlockOptions:
    """The block options of a run."""
    tolerances = Tolerances(fp_rel_tol=float(run.attrs["fp_rel_tol"]),
                            rdf_nrmsd_tol=float(run.attrs["rdf_nrmsd_tol"]),
                            spearman_tol=float(run.attrs["spearman_tol"]), n_stable=int(run.attrs["n_stable"]))
    return BlockOptions(block_ns=float(run.attrs["block_ns"]), tolerances=tolerances)


def run_summary(run: xr.Dataset) -> ConvergenceSummary:
    """What comparing the blocks of a run concluded."""
    options = run_block_options(run)
    return summarize(run, options.block_ns, options.tolerances)


def run_facts(run: xr.Dataset) -> dict:
    """The attributes of a run other than its settings and metadata, as plain Python values."""
    return {name: np.asarray(value).tolist() for name, value in run.attrs.items()
            if name not in [*SETTINGS, "metadata"]}


def run_metadata(run: xr.Dataset) -> dict:
    """The metadata stored with a run (see :func:`oneopes.reports.write_run`): versions, inputs and options."""
    return json.loads(run.attrs["metadata"])


def run_block(run: xr.Dataset, block: int | None = None) -> xr.Dataset:
    """Block ``block`` of a run, by default its converged block; raises InputError when there is none."""
    if block is None:
        block = run_summary(run).converged_block
        if block is None:
            raise InputError("The run did not converge: choose a block.")
    if block not in run.block:
        raise InputError(f"The run has blocks 0 to {run.sizes['block'] - 1}, not {block}.")
    return run.sel(block=block)


def save_run(run: xr.Dataset, path: Path) -> None:
    """Write a block run to a compressed NetCDF file."""
    run.to_netcdf(path, engine="h5netcdf", encoding={name: {"compression": "gzip"} for name in run.data_vars})


def load_run(path: Path) -> xr.Dataset:
    """A block run written by :func:`save_run`."""
    return xr.load_dataset(path, engine="h5netcdf")
