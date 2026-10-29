"""Trajectory blocks for tests that build block runs by hand."""
from oneopes.trajectory_blocks import Block


def one_frame_blocks(n_blocks: int, block_ns: float = 1.0, start_ns: float = 0.0) -> list[Block]:
    """Consecutive blocks of one frame each, ``block_ns`` ns long, from frame 0 at ``start_ns``."""
    return [Block(range(b, b + 1), start_ns + b * block_ns, start_ns + (b + 1) * block_ns) for b in range(n_blocks)]
