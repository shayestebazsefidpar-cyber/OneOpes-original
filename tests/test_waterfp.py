"""
Regression tests for the shared `waterfp` module: the WaterFP analysis must
reproduce the original compute_density_profile() / fp_from_density_profile()
pipeline exactly.
"""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

from waterfp import WaterFP, compute_density_profile, fp_from_density_profile, make_bins

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
        counts = WaterFP(solute, water).run(start=start, stop=stop).results.water_counts
        expected = compute_density_profile(u, solute.indices, water.indices, start, stop, edges_a, shell_vol_nm3)
        assert counts.n_frames == stop - start
        assert np.array_equal(counts.solute_indices, solute.indices)
        assert np.array_equal(counts.density_nm3, expected)


def test_fp_and_g_match_fp_from_density_profile():
    u = random_universe()
    solute, water = u.atoms[:3], u.atoms[3:]
    results = WaterFP(solute, water).run().results
    _edges_nm, centers_nm, edges_a, shell_vol_nm3 = make_bins()
    n_r = compute_density_profile(u, solute.indices, water.indices, 0, len(u.trajectory), edges_a, shell_vol_nm3)
    for i in range(len(solute)):
        fp, g, _norm = fp_from_density_profile(n_r[i], centers_nm, binwidth_nm=0.001)
        assert results.fp[i] == fp
        assert np.array_equal(results.g[i], g)
    assert np.array_equal(results.r_nm, centers_nm)


def test_results_default_to_the_whole_trajectory():
    u = random_universe()
    results = WaterFP(u.atoms[:3], u.atoms[3:]).run().results
    assert results.water_counts.n_frames == len(u.trajectory)
    assert results.water_counts.counts.dtype == np.int64
    assert results.fp.shape == (3,)
    assert results.g.shape == (3, results.water_counts.n_bins)


def test_parallel_run_gives_the_serial_results():
    u = random_universe(n_frames=6)
    solute, water = u.atoms[:3], u.atoms[3:]
    serial = WaterFP(solute, water).run().results
    parallel = WaterFP(solute, water).run(backend="multiprocessing", n_workers=2).results
    assert parallel.water_counts.n_frames == serial.water_counts.n_frames == 6
    assert np.array_equal(parallel.water_counts.counts, serial.water_counts.counts)
    assert np.array_equal(parallel.fp, serial.fp)
