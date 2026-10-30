"""Trajectory blocks for tests that build block runs by hand."""
from collections.abc import Sequence

import numpy as np
import xarray as xr

from oneopes.trajectory_blocks import Block, stack_blocks
from oneopes.waterfp import HydrationOptions, RadialGrid, hydration_from_counts

COARSE_OPTIONS = HydrationOptions(grid=RadialGrid(rmax_nm=1.0, binwidth_nm=0.01), norm_tail_bins=20)
"""A coarse grid, so that synthetic counts are small and quick to draw."""
BULK_DENSITY_NM3 = 33.4
FIRST_SHELL_BINS = 30
"""The bins, up to 0.3 nm on the coarse grid, whose density ``water_blocks`` scales."""


def one_frame_blocks(n_blocks: int, block_ns: float = 1.0, start_ns: float = 0.0) -> list[Block]:
    """Consecutive blocks of one frame each, ``block_ns`` ns long, from frame 0 at ``start_ns``."""
    return [Block(range(b, b + 1), start_ns + b * block_ns, start_ns + (b + 1) * block_ns) for b in range(n_blocks)]


def water_blocks(shell_scales: Sequence[float], n_atoms: int = 2, frames: int = 50, seed: int = 0) -> xr.Dataset:
    """Blocks of 1 ns of Poisson water counts around atoms in bulk water, on the coarse grid.

    In block b the density within 0.3 nm is ``shell_scales[b]`` times bulk, and so is each frame's first-shell count.
    """
    rng = np.random.default_rng(seed)
    bulk_counts = BULK_DENSITY_NM3 * COARSE_OPTIONS.grid.shell_volumes_nm3 * frames
    atoms = [f"A{i}" for i in range(n_atoms)]
    measured = []
    for b, scale in enumerate(shell_scales):
        expected = np.tile(bulk_counts, (n_atoms, 1))
        expected[:, :FIRST_SHELL_BINS] *= scale
        hydration = hydration_from_counts(rng.poisson(expected), frames, atoms, COARSE_OPTIONS)
        shell_count = rng.poisson(expected[:, :FIRST_SHELL_BINS].sum(axis=1) / frames, size=(frames, n_atoms))
        block = Block(range(b * frames, (b + 1) * frames), float(b), float(b + 1))
        measured.append(block.label(hydration.assign(shell_count=(("frame", "atom"), shell_count))))
    return stack_blocks(measured)
