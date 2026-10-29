"""Trajectory blocks: consecutive runs of the same number of frames, each a ``range`` of frame indices."""
import numpy as np
from MDAnalysis import Universe


def frame_step_ps(u: Universe) -> float:
    """Time between frames (ps); raises ValueError unless the frames are evenly spaced."""
    n = len(u.trajectory)
    if n < 2:
        raise ValueError("The trajectory needs at least two frames.")
    t0 = u.trajectory[0].time
    dt = u.trajectory[1].time - t0
    if not np.isclose(u.trajectory[n - 1].time - t0, dt * (n - 1), rtol=1e-6, atol=1e-3):
        raise ValueError("Trajectory frames are not evenly spaced in time.")
    if not (dt > 0 and np.isfinite(dt)):
        raise ValueError(f"Trajectory time step must be > 0, got {dt}.")
    return float(dt)


def frames_per_block(block_ns: float, dt_ps: float) -> int:
    """Frames in a ``block_ns`` block; raises ValueError unless that is a whole, positive number of frames."""
    if not (block_ns > 0 and np.isfinite(block_ns)):
        raise ValueError(f"block_ns must be > 0, got {block_ns}.")
    n = int(round(block_ns * 1000.0 / dt_ps))
    if n < 1 or not np.isclose(n * dt_ps, block_ns * 1000.0, rtol=0, atol=1e-6 * max(1.0, block_ns * 1000.0)):
        raise ValueError(f"block_ns={block_ns} is not a whole number of {dt_ps} ps frames.")
    return n


def frame_time_ns(u: Universe, frame: int) -> float:
    """Simulation time of a frame (ns)."""
    return float(u.trajectory[frame].time) / 1000.0


def time_span_ns(u: Universe, frames: range, dt_ps: float) -> tuple[float, float]:
    """Start and end (ns) of a block of frames: the time of its first frame, and that plus its length."""
    start = frame_time_ns(u, frames.start)
    return start, start + len(frames) * dt_ps / 1000.0


def complete_blocks(start: int, stop: int, length: int) -> list[range]:
    """The complete blocks of ``length`` frames in frames [start, stop); leftover frames at the end are skipped."""
    return [range(first, first + length) for first in range(start, stop - length + 1, length)]
