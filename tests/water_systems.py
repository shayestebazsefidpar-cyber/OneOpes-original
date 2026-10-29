"""Synthetic universes of solute atoms and water oxygens, for tests of the WaterFP analysis."""
import MDAnalysis as mda
import numpy as np
from MDAnalysis.coordinates.memory import MemoryReader

BOX_A = 50.0
"""Cubic box edge (A): half of it (25 A) is more than rmax (20.01 A), so every bin is inside the minimum image."""


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
