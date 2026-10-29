"""A block run: the measured trajectory blocks, each compared with the one before, as one Dataset (blocks.nc).

Its attributes are the settings the comparison used (``block_ns`` and the :class:`Tolerances`) and any
facts about the run, e.g. the ligand selection; the convergence summary is derived from it, not stored.
"""

from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np
import xarray as xr

from oneopes.convergence import ConvergenceSummary, Tolerances, compare_blocks, summarize
from oneopes.trajectory_blocks import stack_blocks

SETTINGS = ["block_ns", *asdict(Tolerances())]
"""The attributes holding the settings of the comparison."""


def block_run(measured: Sequence[xr.Dataset], tol: Tolerances, block_ns: float, **facts) -> xr.Dataset:
    """The measured blocks, stacked and compared (:func:`compare_blocks`), with the settings and ``facts``
    (strings, numbers or lists of them) as attributes."""
    return compare_blocks(stack_blocks(measured), tol).assign_attrs(block_ns=block_ns, **asdict(tol), **facts)


def run_tolerances(run: xr.Dataset) -> Tolerances:
    """The tolerances a block run was compared with."""
    return Tolerances(fp_rel_tol=float(run.attrs["fp_rel_tol"]), rdf_nrmsd_tol=float(run.attrs["rdf_nrmsd_tol"]),
                      spearman_tol=float(run.attrs["spearman_tol"]), n_stable=int(run.attrs["n_stable"]))


def run_summary(run: xr.Dataset) -> ConvergenceSummary:
    """What comparing the blocks of a run concluded."""
    return summarize(run, float(run.attrs["block_ns"]), run_tolerances(run))


def run_facts(run: xr.Dataset) -> dict:
    """The attributes of a run other than its settings, as plain Python values."""
    return {name: np.asarray(value).tolist() for name, value in run.attrs.items() if name not in SETTINGS}


def save_run(run: xr.Dataset, path: Path) -> None:
    """Write a block run to a compressed NetCDF file."""
    run.to_netcdf(path, engine="h5netcdf", encoding={name: {"compression": "gzip"} for name in run.data_vars})


def load_run(path: Path) -> xr.Dataset:
    """A block run written by :func:`save_run`."""
    return xr.load_dataset(path, engine="h5netcdf")
