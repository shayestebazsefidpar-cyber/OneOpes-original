"""A block run: the measured trajectory blocks, each compared with the one before, as one Dataset (blocks.nc).

Its attributes hold the settings and facts of the run; the convergence summary is derived from it, not stored.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

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
    """How long a block is, and when consecutive blocks count as converged."""
    block_ns: Annotated[float, Parameter(validator=Number(gt=0))] = BLOCK_NS_DEFAULT
    """Trajectory block size (ns)."""
    tolerances: Annotated[Tolerances, Parameter(name="*")] = DEFAULT_TOLERANCES


def settings(block_options: BlockOptions, hydration_options: HydrationOptions) -> dict[str, float]:
    """The options a run is made with, as flat attributes."""
    return {"block_ns": block_options.block_ns, **asdict(block_options.tolerances),
            **asdict(hydration_options.grid), "norm_tail_bins": hydration_options.norm_tail_bins}


SETTINGS = list(settings(BlockOptions(), HydrationOptions()))
"""The attributes holding the settings of a run."""


def block_run(measured: Sequence[xr.Dataset], block_options: BlockOptions, hydration_options: HydrationOptions,
              **facts: str | float | list[str]) -> xr.Dataset:
    """The measured blocks, stacked and compared, with the settings and ``facts`` as attributes."""
    compared = compare_blocks(stack_blocks(measured), block_options.tolerances)
    return compared.assign_attrs(**settings(block_options, hydration_options), **facts)


def block_options_of(run: xr.Dataset) -> BlockOptions:
    """The block options a run was made with, rebuilt from its attributes."""
    tolerances = Tolerances(fp_rel_tol=float(run.attrs["fp_rel_tol"]),
                            rdf_nrmsd_tol=float(run.attrs["rdf_nrmsd_tol"]),
                            spearman_tol=float(run.attrs["spearman_tol"]), n_stable=int(run.attrs["n_stable"]))
    return BlockOptions(block_ns=float(run.attrs["block_ns"]), tolerances=tolerances)


def convergence_summary(run: xr.Dataset) -> ConvergenceSummary:
    """What comparing the blocks of a run concluded."""
    options = block_options_of(run)
    return summarize(run, options.block_ns, options.tolerances)


def facts_of(run: xr.Dataset) -> dict[str, Any]:
    """The attributes of a run other than its settings and metadata, as plain Python values."""
    return {name: np.asarray(value).tolist() for name, value in run.attrs.items()
            if name not in [*SETTINGS, "metadata"]}


def metadata_of(run: xr.Dataset) -> dict[str, Any]:
    """The metadata stored with a run: versions, inputs and options."""
    return json.loads(run.attrs["metadata"])


def chosen_block(run: xr.Dataset, block: int | None = None) -> xr.Dataset:
    """Block ``block`` of a run, by default its converged block; raises InputError when there is none."""
    if block is None:
        block = convergence_summary(run).converged_block
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
