"""Where a command's results came from: the package versions and the input files it used."""

import hashlib
import json
import platform
from collections.abc import Mapping
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from oneopes import __version__

PACKAGES = ("numpy", "pandas", "scipy", "xarray", "h5netcdf", "MDAnalysis", "freesasa")
"""The libraries whose versions are recorded."""


def installed_version(package: str) -> str:
    """The installed version of ``package``, or "unknown"."""
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return "unknown"


def versions() -> dict[str, str]:
    """The versions of Python, oneopes and the main libraries."""
    return {"python": platform.python_version(), "oneopes": __version__} | {
        package: installed_version(package) for package in PACKAGES}


def file_record(path: Path) -> dict[str, str | int]:
    """File name, SHA-256 and size of an input file."""
    with path.open("rb") as fh:
        sha256 = hashlib.file_digest(fh, "sha256").hexdigest()
    return {"file": path.name, "sha256": sha256, "bytes": path.stat().st_size}


def provenance(inputs: Mapping[str, Path]) -> dict[str, Any]:
    """The versions in use and a record of each input file, by role (e.g. ``{"tpr": ..., "xtc": ...}``)."""
    return {"versions": versions(), "inputs": {role: file_record(path) for role, path in inputs.items()}}


def json_value(value: range | np.generic) -> list[int] | float | int | bool:
    """JSON stand-in for the values ``json`` cannot write: a range as [start, stop], numpy scalars as Python ones."""
    return [value.start, value.stop] if isinstance(value, range) else value.item()


def to_json(data: Mapping[str, Any]) -> str:
    """``data`` as indented JSON, ranges and numpy scalars included."""
    return json.dumps(data, indent=2, default=json_value)
