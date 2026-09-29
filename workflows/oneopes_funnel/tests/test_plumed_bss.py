"""F / D-3 / D-5 / E-5: the funnel as the real BSS PLUMED writer (Process/_plumed.py) writes it.

The unmodified BSS writer (BioSimSpace.Process.Plumed) is called directly;
the helpers below are test infrastructure only and generate no PLUMED text
themselves. Its constructor only needs a ``plumed``
executable to report a version (>= 2.7 avoids an auxiliary-file branch), so a
test-only stub that answers ``plumed info --version`` is put on PATH; the
writer itself never runs PLUMED. No real PLUMED run is performed here.
"""

import math
import os
import re

import numpy as np
import pytest
from conftest import molecule_layout

from oneopes_funnel.geometry import funnel_parameters


def _upper_bound_nm_that_fits_box(system):
    """Test input only: an upper bound comfortably inside the box (0.9 x half the smallest box length).

    This does not validate anything - whether a bound is acceptable is decided
    by the real BSS writer (_plumed.py:313-362) when the config is created.
    """
    lengths_A = [length.angstroms().value() for length in system.getBox()[0]]
    return min(4.0, math.floor(0.9 * min(lengths_A) / 2) / 10.0)


def _bss_plumed_config(system, cv, tmp_path_factory):
    """Test-only: return what the unmodified BSS PLUMED writer produces for ``cv``.

    Calls ``BioSimSpace.Process.Plumed(...).createConfig()`` directly; no
    PLUMED text is generated here. This is not part of the oneopes_funnel API.
    """
    from BioSimSpace.Process import Plumed
    from BioSimSpace.Protocol import Metadynamics

    bindir = tmp_path_factory.mktemp("stub_plumed_bin")
    stub = bindir / "plumed"
    stub.write_text("#!/bin/sh\necho 2.9\n")
    stub.chmod(0o755)
    workdir = tmp_path_factory.mktemp("plumed_work")
    old_path = os.environ["PATH"]
    os.environ["PATH"] = f"{bindir}{os.pathsep}{old_path}"
    try:
        config, _aux = Plumed(str(workdir)).createConfig(system, Metadynamics(collective_variable=cv))
    finally:
        os.environ["PATH"] = old_path
    return config


def _funnel_with_upper_bound(system, upper_nm):
    """Test-only: make_funnel() with a BSS Bound as upper bound (all other BSS defaults)."""
    from BioSimSpace.Metadynamics import Bound
    from BioSimSpace.Types import Length

    from oneopes_funnel.make_funnel import make_funnel

    return make_funnel(system, upper_bound=Bound(Length(upper_nm, "nanometer"), force_constant=2000))


@pytest.fixture(scope="module")
def bss_plumed(ref, tmp_path_factory):
    from BioSimSpace.Metadynamics.CollectiveVariable import Funnel

    f = _funnel_with_upper_bound(ref.system, _upper_bound_nm_that_fits_box(ref.system))
    assert isinstance(f.cv, Funnel)
    return f, _bss_plumed_config(ref.system, f.cv, tmp_path_factory)


def _line(config, prefix):
    matches = [line.strip() for line in config if line.strip().startswith(prefix)]
    assert len(matches) == 1, (prefix, matches)
    return matches[0]


def _serials(line):
    return [int(x) for x in line.split("ATOMS=")[1].split(",")]


def test_f1_d3_p1_label_is_atoms0_plus_one(bss_plumed):
    f, config = bss_plumed
    assert _serials(_line(config, "p1: COM ATOMS=")) == [i + 1 for i in f.atoms0]


def test_f2_d3_p2_label_is_atoms1_plus_one(bss_plumed):
    f, config = bss_plumed
    assert _serials(_line(config, "p2: COM ATOMS=")) == [i + 1 for i in f.atoms1]


def test_f3_d5_ligand_com_is_the_ligand_molecule_1_based(ref, bss_plumed):
    _, config = bss_plumed
    _, ligand, offsets, counts = molecule_layout(ref.system)
    assert _line(config, "lig: COM ATOMS=") == f"lig: COM ATOMS={offsets[ligand] + 1}-{offsets[ligand] + counts[ligand]}"


def test_f4_projection_on_axis_from_p1_to_p2(bss_plumed):
    assert _line(bss_plumed[1], "pp:") == "pp: PROJECTION_ON_AXIS AXIS_ATOMS=p1,p2 ATOM=lig"


def test_f5_e5_funnel_constants_are_the_cv_parameters_in_nm(bss_plumed):
    f, config = bss_plumed
    p = funnel_parameters(f.cv)
    for label, value in [("s_cent", p.inflection_nm), ("beta_cent", p.steepness_per_nm), ("wall_width", p.width_nm), ("wall_buffer", p.buffer_nm)]:
        assert float(_line(config, f"{label}: CONSTANT VALUES=").split("=")[1]) == value


def test_f6_wall_center_formula_equals_getextent(bss_plumed):
    f, config = bss_plumed
    assert _line(config, "FUNC=h*") == "FUNC=h*(1./(1.+exp(b*(s-sc))))+f"
    p = funnel_parameters(f.cv)
    s = np.array([0.0, 0.5, 1.0, 2.0, 3.0])
    plumed_value = p.width_nm * (1.0 / (1.0 + np.exp(p.steepness_per_nm * (s - p.inflection_nm)))) + p.buffer_nm
    np.testing.assert_allclose(plumed_value, f.extent_nm(s), atol=1e-12)


def test_f7_e5_projection_walls_are_the_cv_bounds_in_nm(bss_plumed):
    f, config = bss_plumed
    p = funnel_parameters(f.cv)
    for label, at, kappa in [("lwall1: LOWER_WALLS", p.lower_bound_nm, p.lower_bound_force_constant), ("uwall1: UPPER_WALLS", p.upper_bound_nm, p.upper_bound_force_constant)]:
        line = _line(config, label)
        assert float(re.search(r"AT=([\d.]+)", line).group(1)) == at
        assert float(re.search(r"KAPPA=([\d.]+)", line).group(1)) == kappa


def test_f8_wholemolecules_protein_then_ligand(ref, bss_plumed):
    protein, ligand, offsets, counts = molecule_layout(ref.system)
    expected = (
        f"WHOLEMOLECULES ENTITY0={offsets[protein] + 1}-{offsets[protein] + counts[protein]}"
        f" ENTITY1={offsets[ligand] + 1}-{offsets[ligand] + counts[ligand]}"
    )
    assert _line(bss_plumed[1], "WHOLEMOLECULES") == expected


def test_f10_bss_labels_are_p1_p2_not_p0_p1(bss_plumed):
    labels = [line.split(":")[0].strip() for line in bss_plumed[1] if re.match(r"^\s*p\d+:", line)]
    assert labels == ["p1", "p2"]  # BSS P0 -> "p1", P1 -> "p2" (production plumed.dat uses p0/p1)


def test_f12_wall_line_format_recorded_as_bss_writes_it(bss_plumed):
    # BSS writes comma-separated keywords (_plumed.py:752-786); recorded, not judged.
    pattern = r"^{}: {} ARG=pp\.proj, AT=[\d.]+, KAPPA=[\d.]+, EXP=[\d.]+, EPS=[\d.]+$"
    assert re.match(pattern.format("lwall1", "LOWER_WALLS"), _line(bss_plumed[1], "lwall1:"))
    assert re.match(pattern.format("uwall1", "UPPER_WALLS"), _line(bss_plumed[1], "uwall1:"))


# --------------------------------------------------------------------------
# Regression: BSS PLUMED writer box check (_plumed.py:313-362) - BSS decides,
# the tests only choose the inputs.
# --------------------------------------------------------------------------
def test_f13_hsp90_production_upper_bound_3p7_nm_is_rejected_by_bss(ref, tmp_path_factory):
    """Production plumed.dat uses uwall AT=3.7 nm; BSS refuses it for the HSP90 box."""
    from BioSimSpace._Exceptions import IncompatibleError

    if "HSP90" not in ref.directory:
        pytest.skip("the explicit 3.7 nm case applies to the HSP90 validation system")
    lengths_A = [length.angstroms().value() for length in ref.system.getBox()[0]]
    assert min(lengths_A) == pytest.approx(73.67, abs=0.01)  # half = 36.84 A < 37 A
    with pytest.raises(IncompatibleError, match="box is too small for the funnel"):
        _bss_plumed_config(ref.system, _funnel_with_upper_bound(ref.system, 3.7).cv, tmp_path_factory)
    # control: 3.6 nm = 36 A (< 36.84 A) is accepted by the same writer
    assert _bss_plumed_config(ref.system, _funnel_with_upper_bound(ref.system, 3.6).cv, tmp_path_factory)


def test_f14_bss_rejects_upper_bound_at_half_the_smallest_box_length(ref, tmp_path_factory):
    from BioSimSpace._Exceptions import IncompatibleError

    half_nm = min(length.angstroms().value() for length in ref.system.getBox()[0]) / 2 / 10
    with pytest.raises(IncompatibleError, match="box is too small for the funnel"):
        _bss_plumed_config(ref.system, _funnel_with_upper_bound(ref.system, half_nm + 0.001).cv, tmp_path_factory)
    assert _bss_plumed_config(ref.system, _funnel_with_upper_bound(ref.system, half_nm - 0.001).cv, tmp_path_factory)
