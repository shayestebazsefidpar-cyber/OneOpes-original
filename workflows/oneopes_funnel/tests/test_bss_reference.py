"""The package relies only on the public BioSimSpace funnel API.

These checks are version-independent: they assert the public API the package
uses exists, and that neither the package nor its tests touch private
(``_``-prefixed) BioSimSpace modules or attributes.
"""

import inspect
import re
from pathlib import Path

import pytest

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
# Built by concatenation so this file does not match its own patterns.
PRIVATE_BSS_PATTERNS = [
    re.compile("BioSim" + r"Space(\.\w+)*\._\w"),  # dotted access to a private module/attribute
    re.compile(r"from\s+BioSim" + r"Space[\w.]*\s+import\s+.*\b_\w"),  # from-import of an underscore name
    re.compile(r"\._sire" + r"_object\b"),  # BioSimSpace wrapper internals
]


def test_no_private_biosimspace_usage_in_package_or_tests():
    hits = []
    for path in sorted(PACKAGE_ROOT.glob("src/**/*.py")) + sorted(PACKAGE_ROOT.glob("tests/*.py")):
        for n, line in enumerate(path.read_text().splitlines(), 1):
            if any(p.search(line) for p in PRIVATE_BSS_PATTERNS):
                hits.append(f"{path.relative_to(PACKAGE_ROOT)}:{n}: {line.strip()}")
    assert hits == []


@pytest.fixture(scope="module")
def bss():
    return pytest.importorskip("BioSimSpace")


def test_public_funnel_api_is_available(bss):
    from BioSimSpace.Metadynamics import Bound
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel, makeFunnel
    from BioSimSpace.Types import Length

    params = inspect.signature(makeFunnel).parameters
    for name in ("system", "protein", "ligand", "alpha_carbon_name", "property_map"):
        assert name in params, name
    for method in ("getAtoms0", "getAtoms1", "getWidth", "getBuffer", "getSteepness", "getInflection",
                   "getLowerBound", "getUpperBound", "getExtent", "getCorrection"):
        assert callable(getattr(Funnel, method)), method
    assert callable(Bound) and callable(Length)


def test_public_extent_contract(bss):
    """getExtent(Length) -> Length; width + buffer deep inside, width/2 + buffer at the inflection."""
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel
    from BioSimSpace.Types import Length

    cv = Funnel([0], [1])
    width = cv.getWidth().nanometers().value()
    buffer = cv.getBuffer().nanometers().value()
    at_inflection = cv.getExtent(cv.getInflection()).nanometers().value()
    assert at_inflection == pytest.approx(width / 2 + buffer, abs=1e-12)
    deep = cv.getExtent(Length(-1000.0, "nanometer")).nanometers().value()
    assert deep == pytest.approx(width + buffer, abs=1e-9)
