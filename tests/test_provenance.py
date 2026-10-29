"""Tests for the `provenance` module: versions, input file records and their JSON."""
import hashlib
import json

import numpy as np
import pytest

from oneopes import __version__
from oneopes.provenance import PACKAGES, file_record, provenance, to_json, versions
from oneopes.trajectory_blocks import Block, FrameWindow


def test_versions_name_python_oneopes_and_the_main_libraries():
    found = versions()
    assert set(found) == {"python", "oneopes", *PACKAGES}
    assert found["oneopes"] == __version__


def test_a_file_record_names_hashes_and_sizes_the_file(tmp_path):
    path = tmp_path / "input.tpr"
    path.write_bytes(b"topology")
    assert file_record(path) == {"file": "input.tpr", "sha256": hashlib.sha256(b"topology").hexdigest(), "bytes": 8}


def test_provenance_records_each_input_by_role(tmp_path):
    (tmp_path / "a.tpr").write_bytes(b"a")
    (tmp_path / "b.xtc").write_bytes(b"bb")
    record = provenance({"tpr": tmp_path / "a.tpr", "xtc": tmp_path / "b.xtc"})
    assert record["versions"] == versions()
    assert {role: entry["file"] for role, entry in record["inputs"].items()} == {"tpr": "a.tpr", "xtc": "b.xtc"}


def test_to_json_writes_ranges_and_numpy_scalars():
    data = {"frames": range(3, 9), "fp": np.float64(-1.5), "n": np.int64(4), "ok": np.bool_(True)}
    assert json.loads(to_json(data)) == {"frames": [3, 9], "fp": -1.5, "n": 4, "ok": True}


def test_to_json_writes_dataclasses_as_dicts():
    data = {"window": FrameWindow(first_frame=2), "block": Block(range(5), 0.0, 5.0)}
    assert json.loads(to_json(data)) == {"window": {"first_frame": 2, "last_frame": None},
                                         "block": {"frames": [0, 5], "start_ns": 0.0, "end_ns": 5.0}}


def test_to_json_rejects_what_it_cannot_write():
    with pytest.raises(TypeError, match="set cannot be written as JSON"):
        to_json({"atoms": {"C1"}})
