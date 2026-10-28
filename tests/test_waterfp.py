"""
Regression tests for the shared `waterfp` module: the new counting path
(WaterCountsAnalysis -> WaterCounts.density_nm3) must reproduce the original
compute_density_profile() exactly.
"""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from waterfp import WaterCountsAnalysis, compute_density_profile, make_bins

BOX_A = 50.0  # half box (25 A) > rmax (20.01 A): every bin is fully inside the minimum image


def random_universe(n_solute: int = 3, n_water: int = 200, n_frames: int = 5, seed: int = 0) -> mda.Universe:
    """Atoms 0..n_solute-1 are the solute, the rest water oxygens, all at random positions."""
    rng = np.random.default_rng(seed)
    n_atoms = n_solute + n_water
    u = mda.Universe.empty(n_atoms, trajectory=True)
    coords = rng.uniform(0.0, BOX_A, (n_frames, n_atoms, 3)).astype(np.float32)
    u.load_new(coords, format=MemoryReader, dimensions=[BOX_A, BOX_A, BOX_A, 90.0, 90.0, 90.0])
    return u


def test_water_counts_density_matches_compute_density_profile():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    _edges_nm, _centers_nm, edges_a, shell_vol_nm3 = make_bins()
    for start, stop in [(0, 5), (1, 4)]:
        counts = WaterCountsAnalysis(solute, water).run(start=start, stop=stop).results.water_counts
        expected = compute_density_profile(u, solute.indices, water.indices, start, stop, edges_a, shell_vol_nm3)
        assert counts.n_frames == stop - start
        assert np.array_equal(counts.solute_indices, solute.indices)
        assert np.array_equal(counts.density_nm3, expected)


def test_water_counts_default_to_the_whole_trajectory():
    u = random_universe()
    counts = WaterCountsAnalysis(u.atoms[:3], u.atoms[3:]).run().results.water_counts
    assert counts.n_frames == len(u.trajectory)
    assert counts.counts.dtype == np.int64
    assert counts.counts.shape == (3, counts.n_bins)


def test_parallel_run_gives_the_serial_counts():
    u = random_universe(n_frames=6)
    solute, water = u.atoms[:3], u.atoms[3:]
    serial = WaterCountsAnalysis(solute, water).run().results.water_counts
    parallel = WaterCountsAnalysis(solute, water).run(backend="multiprocessing", n_workers=2).results.water_counts
    assert parallel.n_frames == serial.n_frames == 6
    assert np.array_equal(parallel.counts, serial.counts)
