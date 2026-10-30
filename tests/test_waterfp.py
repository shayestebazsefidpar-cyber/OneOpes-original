"""
Tests for the shared `waterfp` module: the radial grid, the counting done by the WaterFP
analysis, and the water structure and FP integral in hydration_from_counts().
"""
import numpy as np
import pytest
import xarray as xr

from oneopes.errors import InputError
from oneopes.waterfp import (
    HydrationOptions,
    RadialGrid,
    WaterFP,
    hydration_from_counts,
    hydration_from_density,
    measure_hydration,
)
from water_systems import random_universe, universe_from_coordinates

BULK_DENSITY_NM3 = 33.4  # water-O number density of bulk water
BULK_FRAMES = 10**12  # many frames, so integer counts follow the bulk density closely even in tiny shells


def bulk_water_counts(n_atoms: int = 1) -> np.ndarray:
    """Counts over BULK_FRAMES frames of a uniform bulk-water density around every atom (g(r) == 1 up to rounding)."""
    counts = np.round(BULK_DENSITY_NM3 * BULK_FRAMES * RadialGrid().shell_volumes_nm3).astype(np.int64)
    return np.tile(counts, (n_atoms, 1))


def test_default_grid_matches_the_waterfp_reference_settings():
    grid = RadialGrid()
    assert grid.n_bins == 2001
    assert len(grid.edges_nm) == 2002
    assert grid.edges_nm[0] == 0.0
    assert np.isclose(grid.edges_nm[-1], 2.001)
    assert np.allclose(grid.r_nm, grid.edges_nm[:-1] + 0.0005)
    assert np.array_equal(grid.edges_A, 10.0 * grid.edges_nm)


def test_custom_grid_has_uniform_bins_and_growing_shells():
    grid = RadialGrid(rmax_nm=1.0, binwidth_nm=0.1)
    assert grid.n_bins == 10
    assert np.allclose(np.diff(grid.edges_nm), 0.1)
    assert np.all(grid.shell_volumes_nm3 > 0)
    assert np.all(np.diff(grid.shell_volumes_nm3) > 0)


def test_density_is_counts_per_frame_per_shell_volume():
    grid = RadialGrid()
    raw = np.arange(2 * 2001, dtype=np.int64).reshape(2, 2001)
    density = hydration_from_counts(raw, 4, ["a", "b"]).density
    assert np.array_equal(density.values, raw / 4 / grid.shell_volumes_nm3)


def test_hydration_is_indexed_by_atom_name_and_radius():
    result = hydration_from_counts(bulk_water_counts(2), BULK_FRAMES, ["C1", "O1"])
    assert list(result.atom.values) == ["C1", "O1"]
    assert np.array_equal(result.r_nm, RadialGrid().r_nm)
    assert result.fp.dims == ("atom",)
    assert result.g.dims == ("atom", "r_nm")


def test_a_water_is_counted_in_the_bin_of_its_distance():
    # solute at the box centre; one water 0.50025 nm away along x -> bin 500 ([0.500, 0.501) nm),
    # one 1.80025 nm away along y -> bin 1800 (in the bulk tail, so the FP can be normalised)
    coords = np.array([[[25.0, 25.0, 25.0], [30.0025, 25.0, 25.0], [25.0, 43.0025, 25.0]]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.counts
    assert counts.sum() == 2
    assert counts[0, 500] == 1
    assert counts[0, 1800] == 1


def test_waters_beyond_rmax_are_not_counted():
    # one water 20.5 A away (> rmax = 20.01 A) and one 18.0025 A away, in the bulk tail (bin 1800)
    coords = np.array([[[5.0, 5.0, 5.0], [5.0, 5.0, 25.5], [5.0, 23.0025, 5.0]]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.counts
    assert counts.sum() == 1
    assert counts[0, 1800] == 1


def test_no_water_in_the_bulk_tail_is_an_error():
    with pytest.raises(InputError, match=r"No water in the bulk tail .* atoms \['C1', 'N1'\]"):
        hydration_from_counts(np.zeros((2, 2001), dtype=np.int64), 1, ["C1", "N1"])


@pytest.mark.parametrize("rmax_nm, binwidth_nm", [(2.0, 0.0), (2.0, -0.1), (0.5, 1.0), (0.0, 0.001)])
def test_radial_grid_needs_a_positive_bin_within_rmax(rmax_nm, binwidth_nm):
    with pytest.raises(InputError):
        RadialGrid(rmax_nm=rmax_nm, binwidth_nm=binwidth_nm)


@pytest.mark.parametrize("norm_tail_bins", [0, 2002])
def test_bulk_tail_must_lie_within_the_grid(norm_tail_bins):
    with pytest.raises(InputError, match="norm_tail_bins"):
        HydrationOptions(norm_tail_bins=norm_tail_bins)


def test_rmax_beyond_half_the_box_warns():
    u = random_universe(box_a=30.0)  # half box 15 A < rmax 20.01 A
    with pytest.warns(RuntimeWarning, match="larger than half the shortest box edge"):
        WaterFP(u.atoms[:3], u.atoms[3:]).run()


def test_counts_of_consecutive_frame_ranges_add_up():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    whole = WaterFP(solute, water).run().results.counts
    first = WaterFP(solute, water).run(stop=2).results.counts
    rest = WaterFP(solute, water).run(start=2).results.counts
    assert np.array_equal(first + rest, whole)


def test_fp_is_zero_for_uniform_bulk_density():
    result = hydration_from_counts(bulk_water_counts(), BULK_FRAMES, ["C1"])
    assert np.allclose(result.g, 1.0, atol=1e-4)
    assert abs(float(result.fp[0])) < 1e-6


def test_fp_is_negative_for_a_depleted_shell():
    # bulk everywhere except a fully depleted (g = 0) first shell: the g -> 0
    # limit of (g ln g - g + 1) is 1, so the shell contributes a negative FP.
    depleted = bulk_water_counts()
    depleted[:, :50] = 0
    result = hydration_from_counts(depleted, BULK_FRAMES, ["C1"])
    assert np.all(result.g[0, :50] == 0.0)
    assert float(result.fp[0]) < 0


def test_measure_hydration_names_atoms_by_label_or_atom_name():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    solute.names = ["C1", "N1", "O1"]
    frames = range(len(u.trajectory))
    assert list(measure_hydration(solute, water, frames).atom.values) == ["C1", "N1", "O1"]
    assert list(measure_hydration(solute, water, frames, labels=["a", "b", "c"]).atom.values) == ["a", "b", "c"]


def test_measure_hydration_covers_the_given_frames():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    counts = WaterFP(solute, water).run(start=1, stop=3).results.counts
    expected = hydration_from_counts(counts, 2, list(solute.names)).assign_coords(atom_index=("atom", [0, 1, 2]))
    measured = measure_hydration(solute, water, range(1, 3))
    assert measured.drop_vars("shell_count").identical(expected)
    assert measured.shell_count.sizes == {"frame": 2, "atom": 3}


def test_the_first_shell_count_of_each_frame_is_recorded():
    """A water 0.3 nm from the atom, then 0.4 nm: in the first shell (0.35 nm), then not; a bulk water stays out."""
    near, far = [28.0, 25.0, 25.0], [29.0, 25.0, 25.0]
    bulk = [25.0, 43.0025, 25.0]
    coords = np.array([[[25.0, 25.0, 25.0], near, bulk], [[25.0, 25.0, 25.0], far, bulk]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    assert WaterFP(u.atoms[:1], u.atoms[1:]).run().results.shell_counts.tolist() == [[1], [0]]


def test_hydration_from_density_works_along_further_dimensions():
    """A density per block gives each block the hydration it has alone."""
    counts = [bulk_water_counts(), bulk_water_counts()]
    counts[1][:, :50] = 0
    alone = [hydration_from_counts(block_counts, BULK_FRAMES, ["C1"]) for block_counts in counts]
    together = hydration_from_density(xr.concat([hydration.density for hydration in alone], dim="block"))
    assert np.allclose(together.fp.values, [hydration.fp.values for hydration in alone])


def test_a_wrong_number_of_labels_is_rejected():
    u = random_universe()
    with pytest.raises(ValueError, match="2 labels for 3 solute atoms"):
        WaterFP(u.atoms[:3], u.atoms[3:], labels=["a", "b"])


def test_parallel_run_gives_the_serial_results():
    u = random_universe(n_frames=6)
    solute, water = u.atoms[:3], u.atoms[3:]
    serial = WaterFP(solute, water).run().results
    parallel = WaterFP(solute, water).run(backend="multiprocessing", n_workers=2).results
    assert np.array_equal(parallel.counts, serial.counts)
    assert parallel.hydration.identical(serial.hydration)
