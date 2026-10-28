"""
Tests for the shared `waterfp` module: the radial grid of WaterCounts, the
counting done by the WaterFP analysis, and the FP integral in fingerprint().
"""
import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

from waterfp import RadialGrid, WaterCounts, WaterFP, fingerprint

BOX_A = 50.0  # half box (25 A) > rmax (20.01 A): every bin is fully inside the minimum image
BULK_DENSITY_NM3 = 33.4  # water-O number density of bulk water


def random_universe(n_solute: int = 3, n_water: int = 200, n_frames: int = 5, seed: int = 0,
                    box_a: float = BOX_A) -> mda.Universe:
    """Atoms 0..n_solute-1 are the solute, the rest water oxygens, all at random positions."""
    rng = np.random.default_rng(seed)
    n_atoms = n_solute + n_water
    coords = rng.uniform(0.0, box_a, (n_frames, n_atoms, 3)).astype(np.float32)
    return universe_from_coordinates(coords, box_a)


def universe_from_coordinates(coords: np.ndarray, box_a: float = BOX_A) -> mda.Universe:
    """Universe with one frame per entry of ``coords`` (n_frames, n_atoms, 3), in a cubic box."""
    u = mda.Universe.empty(coords.shape[1], trajectory=True)
    u.load_new(coords, format=MemoryReader, dimensions=[box_a, box_a, box_a, 90.0, 90.0, 90.0])
    return u


def water_counts(counts: np.ndarray, n_frames: int = 1) -> WaterCounts:
    """WaterCounts on the default grid, for atoms 0..len(counts)-1."""
    return WaterCounts(counts=counts, n_frames=n_frames, solute_indices=np.arange(len(counts)), grid=RadialGrid())


def bulk_water_counts(n_atoms: int = 1) -> WaterCounts:
    """Counts of a uniform bulk-water density around every atom (g(r) == 1 up to integer rounding)."""
    n_frames = 10**12  # many frames, so integer counts follow the bulk density closely even in tiny shells
    counts = np.round(BULK_DENSITY_NM3 * n_frames * RadialGrid().shell_volumes_nm3).astype(np.int64)
    return water_counts(np.tile(counts, (n_atoms, 1)), n_frames=n_frames)


def test_default_grid_matches_the_waterfp_reference_settings():
    grid = RadialGrid()
    assert grid.n_bins == 2001
    assert len(grid.edges_nm) == 2002
    assert grid.edges_nm[0] == 0.0
    assert np.isclose(grid.edges_nm[-1], 2.001)
    assert np.allclose(grid.r_nm, grid.edges_nm[:-1] + 0.0005)
    assert np.array_equal(grid.edges_a, 10.0 * grid.edges_nm)


def test_custom_grid_has_uniform_bins_and_growing_shells():
    grid = RadialGrid(rmax_nm=1.0, binwidth_nm=0.1)
    assert grid.n_bins == 10
    assert np.allclose(np.diff(grid.edges_nm), 0.1)
    assert np.all(grid.shell_volumes_nm3 > 0)
    assert np.all(np.diff(grid.shell_volumes_nm3) > 0)


def test_density_is_counts_per_frame_per_shell_volume():
    raw = np.arange(2 * 2001, dtype=np.int64).reshape(2, 2001)
    counts = water_counts(raw, n_frames=4)
    assert np.array_equal(counts.density_nm3, raw / 4 / counts.grid.shell_volumes_nm3)


def test_a_water_is_counted_in_the_bin_of_its_distance():
    # solute at the box centre; one water 0.50025 nm away along x -> bin 500 ([0.500, 0.501) nm),
    # one 1.80025 nm away along y -> bin 1800 (in the bulk tail, so the FP can be normalised)
    coords = np.array([[[25.0, 25.0, 25.0], [30.0025, 25.0, 25.0], [25.0, 43.0025, 25.0]]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.water_counts
    assert counts.counts.sum() == 2
    assert counts.counts[0, 500] == 1
    assert counts.counts[0, 1800] == 1


def test_waters_beyond_rmax_are_not_counted():
    # one water 20.5 A away (> rmax = 20.01 A) and one 18.0025 A away, in the bulk tail (bin 1800)
    coords = np.array([[[5.0, 5.0, 5.0], [5.0, 5.0, 25.5], [5.0, 23.0025, 5.0]]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.water_counts
    assert counts.counts.sum() == 1
    assert counts.counts[0, 1800] == 1


def test_no_water_in_the_bulk_tail_is_an_error():
    counts = water_counts(np.zeros((2, 2001), dtype=np.int64))
    with pytest.raises(ValueError, match=r"No water in the bulk tail .* solute atoms \[0, 1\]"):
        fingerprint(counts)


def test_rmax_beyond_half_the_box_warns():
    u = random_universe(box_a=30.0)  # half box 15 A < rmax 20.01 A
    with pytest.warns(RuntimeWarning, match="larger than half the shortest box edge"):
        WaterFP(u.atoms[:3], u.atoms[3:]).run()


def test_counts_of_consecutive_frame_ranges_add_up():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    whole = WaterFP(solute, water).run().results.water_counts
    first = WaterFP(solute, water).run(stop=2).results.water_counts
    rest = WaterFP(solute, water).run(start=2).results.water_counts
    assert (first.n_frames, rest.n_frames, whole.n_frames) == (2, 3, 5)
    assert np.array_equal(first.counts + rest.counts, whole.counts)
    assert np.array_equal(whole.solute_indices, solute.indices)


def test_fp_is_zero_for_uniform_bulk_density():
    fp, g = fingerprint(bulk_water_counts())
    assert np.allclose(g, 1.0, atol=1e-4)
    assert abs(fp[0]) < 1e-6


def test_fp_is_negative_for_a_depleted_shell():
    # bulk everywhere except a fully depleted (g = 0) first shell: the g -> 0
    # limit of (g ln g - g + 1) is 1, so the shell contributes a negative FP.
    bulk = bulk_water_counts()
    depleted = bulk.counts.copy()
    depleted[:, :50] = 0
    fp, g = fingerprint(water_counts(depleted, n_frames=bulk.n_frames))
    assert np.all(g[0, :50] == 0.0)
    assert fp[0] < 0


def test_results_default_to_the_whole_trajectory():
    u = random_universe()
    results = WaterFP(u.atoms[:3], u.atoms[3:]).run().results
    assert results.water_counts.n_frames == len(u.trajectory)
    assert results.water_counts.counts.dtype == np.int64
    assert results.fp.shape == (3,)
    assert results.g.shape == (3, RadialGrid().n_bins)
    assert np.array_equal(results.r_nm, RadialGrid().r_nm)


def test_fp_table_has_one_row_per_atom_named_by_label():
    u = random_universe()
    waterfp = WaterFP(u.atoms[:3], u.atoms[3:]).run()
    table = waterfp.fp_table(["a", "b", "c"])
    assert list(table.columns) == ["atom", "FP"]
    assert list(table["atom"]) == ["a", "b", "c"]
    assert np.array_equal(table["FP"], waterfp.results.fp)


def test_fp_table_labels_default_to_the_solute_atom_names():
    u = random_universe()
    u.add_TopologyAttr("names", ["C1", "N1", "O1"] + ["OW"] * 200)
    assert list(WaterFP(u.atoms[:3], u.atoms[3:]).run().fp_table()["atom"]) == ["C1", "N1", "O1"]


def test_profile_table_has_one_row_per_atom_and_radius():
    u = random_universe()
    waterfp = WaterFP(u.atoms[:3], u.atoms[3:]).run()
    results = waterfp.results
    table = waterfp.profile_table(["a", "b", "c"])
    n_bins = len(results.r_nm)
    assert list(table.columns) == ["atom", "r_nm", "n_r", "g_r"]
    assert len(table) == 3 * n_bins
    second_atom = table[table["atom"] == "b"]
    assert np.array_equal(second_atom["r_nm"], results.r_nm)
    assert np.array_equal(second_atom["n_r"], results.water_counts.density_nm3[1])
    assert np.array_equal(second_atom["g_r"], results.g[1])


def test_tables_reject_a_wrong_number_of_labels():
    u = random_universe()
    waterfp = WaterFP(u.atoms[:3], u.atoms[3:]).run()
    with pytest.raises(ValueError, match="2 labels for 3 solute atoms"):
        waterfp.fp_table(["a", "b"])


def test_parallel_run_gives_the_serial_results():
    u = random_universe(n_frames=6)
    solute, water = u.atoms[:3], u.atoms[3:]
    serial = WaterFP(solute, water).run().results
    parallel = WaterFP(solute, water).run(backend="multiprocessing", n_workers=2).results
    assert parallel.water_counts.n_frames == serial.water_counts.n_frames == 6
    assert np.array_equal(parallel.water_counts.counts, serial.water_counts.counts)
    assert np.array_equal(parallel.fp, serial.fp)
