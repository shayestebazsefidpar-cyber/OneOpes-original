"""A: the installed BioSimSpace is exactly the reviewed reference (2024.4.1).

A version or source mismatch FAILS (it is never skipped): the comparisons in
this suite are only meaningful against the reviewed BSS source.
"""

import hashlib
from pathlib import Path

import pytest
from conftest import EXPECTED_BSS_VERSION

# SHA-256 of the reviewed BioSimSpace 2024.4.1 source files.
REVIEWED_SHA256 = {
    "Metadynamics/CollectiveVariable/_funnel.py": "40f05e1a924a30a99d548f66087198b7953528ede9557766e1e9521ba44388e8",
    "Process/_plumed.py": "0710e759e9d690d362d3dadb4b62dccf6a6fc64f40bb5bf0b8b8064aeb106ae3",
}


@pytest.fixture(scope="module")
def bss():
    return pytest.importorskip("BioSimSpace")


def test_a1_bss_version_is_reviewed_version(bss):
    assert bss.__version__ == EXPECTED_BSS_VERSION, (
        f"BioSimSpace {bss.__version__} installed; this suite validates against {EXPECTED_BSS_VERSION}. "
        "Re-review the BSS funnel source before updating EXPECTED_BSS_VERSION."
    )


@pytest.mark.parametrize("relpath", sorted(REVIEWED_SHA256))
def test_a2_a3_bss_source_unchanged(bss, relpath):
    path = Path(bss.__file__).parent / relpath
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == REVIEWED_SHA256[relpath], f"{path} differs from the reviewed BSS source."


def test_a4_public_bss_funnel_api_importable(bss):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel, makeFunnel, viewFunnel

    assert callable(makeFunnel) and callable(viewFunnel) and isinstance(Funnel, type)
