"""A block run: the measured trajectory blocks, each with the convergence check after it, as one Dataset (blocks.nc).

Its attributes hold the settings and facts of the run; the convergence summary and the FP estimates
are derived from it, not stored.
"""

import json
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import xarray as xr
from cyclopts import Parameter

from oneopes.convergence import (
    DEFAULT_CRITERIA,
    ConvergenceCriteria,
    ConvergenceSummary,
    PrecisionGoal,
    production_estimate,
    summarize,
    with_checks,
)
from oneopes.errors import InputError
from oneopes.trajectory_blocks import BLOCK_NS_DEFAULT
from oneopes.waterfp import HydrationOptions, RadialGrid


@dataclass(frozen=True)
class BlockOptions:
    """How long a block is, and when the checks after the blocks mean converged."""
    block_ns: float = BLOCK_NS_DEFAULT
    """Trajectory block size (ns): the unit of the jackknife, and how often the run is checked."""
    criteria: Annotated[ConvergenceCriteria, Parameter(name="*")] = DEFAULT_CRITERIA

    def __post_init__(self) -> None:
        if not self.block_ns > 0:
            raise InputError(f"block_ns must be > 0, got {self.block_ns}.")


def settings(block_options: BlockOptions, hydration_options: HydrationOptions) -> dict[str, float]:
    """The options a run is made with, as flat attributes."""
    return {"block_ns": block_options.block_ns, **asdict(block_options.criteria),
            **asdict(hydration_options.grid), "norm_tail_bins": hydration_options.norm_tail_bins}


SETTINGS = list(settings(BlockOptions(), HydrationOptions()))
"""The attributes holding the settings of a run."""


def block_run(blocks: xr.Dataset, checks: Sequence[xr.Dataset], block_options: BlockOptions,
              hydration_options: HydrationOptions, goal: PrecisionGoal, **facts: str | float | list[str]) -> xr.Dataset:
    """The stacked ``blocks`` with the check after each, and the settings, the goal and ``facts`` as attributes."""
    return with_checks(blocks, checks).assign_attrs(**settings(block_options, hydration_options),
                                                    goal=goal.description, **facts)


def block_options_of(run: xr.Dataset) -> BlockOptions:
    """The block options a run was made with, rebuilt from its attributes."""
    criteria = ConvergenceCriteria(min_blocks=int(run.attrs["min_blocks"]),
                                   drift_alpha=float(run.attrs["drift_alpha"]),
                                   n_passes=int(run.attrs["n_passes"]))
    return BlockOptions(block_ns=float(run.attrs["block_ns"]), criteria=criteria)


def hydration_options_of(run: xr.Dataset) -> HydrationOptions:
    """The hydration options a run was made with, rebuilt from its attributes."""
    grid = RadialGrid(rmax_nm=float(run.attrs["rmax_nm"]), binwidth_nm=float(run.attrs["binwidth_nm"]))
    return HydrationOptions(grid=grid, norm_tail_bins=int(run.attrs["norm_tail_bins"]))


def convergence_summary(run: xr.Dataset) -> ConvergenceSummary:
    """What the checks of a run concluded."""
    options = block_options_of(run)
    return summarize(run, options.block_ns, options.criteria)


def facts_of(run: xr.Dataset) -> dict[str, Any]:
    """The attributes of a run other than its settings and metadata, as plain Python values."""
    return {name: np.asarray(value).tolist() for name, value in run.attrs.items()
            if name not in [*SETTINGS, "metadata"]}


def metadata_of(run: xr.Dataset) -> dict[str, Any]:
    """The metadata stored with a run: versions, inputs and options."""
    return json.loads(run.attrs["metadata"])


def chosen_block(run: xr.Dataset, block: int | None = None) -> int:
    """Block ``block`` of a run, by default its converged block; raises InputError when there is none."""
    if block is None:
        block = convergence_summary(run).converged_block
        if block is None:
            raise InputError("The run did not converge: choose a block.")
    if block not in run.block:
        raise InputError(f"The run has blocks 0 to {run.sizes['block'] - 1}, not {block}.")
    return block


def production_at(run: xr.Dataset, block: int) -> xr.Dataset:
    """The production blocks as the check after ``block`` found them; raises InputError if that block has no check."""
    start = run.production_start_block.sel(block=block)
    if np.isnan(start):
        raise InputError(f"Block {block} has no check: production needs at least two blocks.")
    return run.sel(block=slice(int(start), block))


def estimate_at(run: xr.Dataset, block: int) -> xr.Dataset:
    """FP and hydration of the production blocks as the check after ``block`` found them."""
    return production_estimate(production_at(run, block), hydration_options_of(run))


def save_run(run: xr.Dataset, path: Path) -> None:
    """Write a block run to a compressed NetCDF file."""
    run.to_netcdf(path, engine="h5netcdf", encoding={name: {"compression": "gzip"} for name in run.data_vars})


def load_run(path: Path) -> xr.Dataset:
    """A block run written by :func:`save_run`."""
    return xr.load_dataset(path, engine="h5netcdf")
