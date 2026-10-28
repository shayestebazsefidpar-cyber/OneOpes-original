"""
Tests for the shared `waterfp` module: the radial grid of WaterCounts, the
counting done by the WaterFP analysis, and the FP integral in fingerprint().
"""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from waterfp import WaterCounts, WaterFP, fingerprint

BOX_A = 50.0  # half box (25 A) > rmax (20.01 A): every bin is fully inside the minimum image
BULK_DENSITY_NM3 = 33.4  # water-O number density of bulk water


def random_universe(n_solute: int = 3, n_water: int = 200, n_frames: int = 5, seed: int = 0) -> mda.Universe:
    """Atoms 0..n_solute-1 are the solute, the rest water oxygens, all at random positions."""
    rng = np.random.default_rng(seed)
    n_atoms = n_solute + n_water
    coords = rng.uniform(0.0, BOX_A, (n_frames, n_atoms, 3)).astype(np.float32)
    return universe_from_coordinates(coords)


def universe_from_coordinates(coords: np.ndarray) -> mda.Universe:
    """Universe with one frame per entry of ``coords`` (n_frames, n_atoms, 3), in a BOX_A cube."""
    u = mda.Universe.empty(coords.shape[1], trajectory=True)
    u.load_new(coords, format=MemoryReader, dimensions=[BOX_A, BOX_A, BOX_A, 90.0, 90.0, 90.0])
    return u


def water_counts(counts: np.ndarray, n_frames: int = 1, rmax_nm: float = 2.001,
                 binwidth_nm: float = 0.001) -> WaterCounts:
    return WaterCounts(counts=counts, n_frames=n_frames, solute_indices=np.arange(len(counts)),
                       rmax_nm=rmax_nm, binwidth_nm=binwidth_nm)


def bulk_water_counts(n_atoms: int = 1) -> WaterCounts:
    """Counts of a uniform bulk-water density around every atom (g(r) == 1 up to integer rounding)."""
    n_frames = 10**12  # many frames, so integer counts follow the bulk density closely even in tiny shells
    template = water_counts(np.zeros((n_atoms, 2001), dtype=np.int64))
    counts = np.round(BULK_DENSITY_NM3 * n_frames * template.shell_volumes_nm3).astype(np.int64)
    return water_counts(np.tile(counts, (n_atoms, 1)), n_frames=n_frames)


def test_default_grid_matches_the_waterfp_reference_settings():
    counts = water_counts(np.zeros((1, 2001), dtype=np.int64))
    assert counts.n_bins == 2001
    assert len(counts.edges_nm) == 2002
    assert counts.edges_nm[0] == 0.0
    assert np.isclose(counts.edges_nm[-1], 2.001)
    assert np.allclose(counts.r_nm, counts.edges_nm[:-1] + 0.0005)


def test_custom_grid_has_uniform_bins_and_growing_shells():
    counts = water_counts(np.zeros((1, 10), dtype=np.int64), rmax_nm=1.0, binwidth_nm=0.1)
    assert counts.n_bins == 10
    assert np.allclose(np.diff(counts.edges_nm), 0.1)
    assert np.all(counts.shell_volumes_nm3 > 0)
    assert np.all(np.diff(counts.shell_volumes_nm3) > 0)


def test_density_is_counts_per_frame_per_shell_volume():
    raw = np.arange(2 * 2001, dtype=np.int64).reshape(2, 2001)
    counts = water_counts(raw, n_frames=4)
    assert np.array_equal(counts.density_nm3, raw / 4 / counts.shell_volumes_nm3)


def test_a_water_is_counted_in_the_bin_of_its_distance():
    # solute at the box centre, one water 0.50025 nm away along x -> bin 500 ([0.500, 0.501) nm)
    coords = np.array([[[25.0, 25.0, 25.0], [30.0025, 25.0, 25.0]]], dtype=np.float32)
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.water_counts
    assert counts.counts.sum() == 1
    assert counts.counts[0, 500] == 1


def test_waters_beyond_rmax_are_not_counted():
    coords = np.array([[[5.0, 5.0, 5.0], [5.0, 5.0, 25.5]]], dtype=np.float32)  # 20.5 A > 20.01 A
    u = universe_from_coordinates(coords)
    counts = WaterFP(u.atoms[:1], u.atoms[1:]).run().results.water_counts
    assert counts.counts.sum() == 0


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
    assert results.g.shape == (3, results.water_counts.n_bins)
    assert np.array_equal(results.r_nm, results.water_counts.r_nm)


def test_parallel_run_gives_the_serial_results():
    u = random_universe(n_frames=6)
    solute, water = u.atoms[:3], u.atoms[3:]
    serial = WaterFP(solute, water).run().results
    parallel = WaterFP(solute, water).run(backend="multiprocessing", n_workers=2).results
    assert parallel.water_counts.n_frames == serial.water_counts.n_frames == 6
    assert np.array_equal(parallel.water_counts.counts, serial.water_counts.counts)
    assert np.array_equal(parallel.fp, serial.fp)
