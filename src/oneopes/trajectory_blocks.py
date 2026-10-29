"""Trajectory blocks: consecutive runs of the same number of frames, each a ``range`` of frame indices."""
from dataclasses import dataclass

import numpy as np
from MDAnalysis import Universe

BLOCK_NS_DEFAULT = 5.0


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
    n = round(block_ns * 1000.0 / dt_ps)
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


@dataclass(frozen=True)
class FrameLayout:
    """The analysed frames of a trajectory, the complete blocks in them, and their times (ns)."""
    n_frames: int
    """Frames in the whole trajectory."""
    dt_ps: float
    analysed: range
    blocks: list[range]
    block_times_ns: list[tuple[float, float]]
    """(start, end) of each block."""
    trajectory_start_ns: float
    trajectory_end_ns: float
    """Time of the trajectory's last frame."""
    analysis_start_ns: float
    analysis_end_ns: float
    """Time of the last analysed frame."""

    @property
    def analysis_time_ns(self) -> float:
        return self.analysis_end_ns - self.analysis_start_ns

    @property
    def tail_frames(self) -> int:
        """Analysed frames after the last complete block, in no block."""
        return self.analysed.stop - self.blocks[-1].stop

    def describe(self) -> str:
        """The layout in one line, for the log."""
        return (f"{self.n_frames} frames of {self.dt_ps} ps "
                f"({self.trajectory_start_ns:g}-{self.trajectory_end_ns:g} ns); "
                f"analysed frames {self.analysed.start}-{self.analysed.stop - 1} ({self.analysis_time_ns:g} ns) "
                f"in {len(self.blocks)} blocks of {len(self.blocks[0])} frames, {self.tail_frames} left over")


def frame_layout(u: Universe, block_ns: float, first_frame: int = 0, last_frame: int | None = None) -> FrameLayout:
    """Split frames [first_frame, last_frame) of ``u`` (default: all) into complete ``block_ns`` blocks.

    Raises ValueError for an invalid frame range, unevenly spaced frames, or fewer frames than one block.
    """
    n_frames = len(u.trajectory)
    analysed = range(first_frame, n_frames if last_frame is None else last_frame)
    if not 0 <= analysed.start < analysed.stop <= n_frames:
        raise ValueError(f"Invalid frame range [{analysed.start}, {analysed.stop}) for a {n_frames}-frame trajectory.")
    dt_ps = frame_step_ps(u)
    blocks = complete_blocks(analysed.start, analysed.stop, frames_per_block(block_ns, dt_ps))
    if not blocks:
        raise ValueError(f"Frames [{analysed.start}, {analysed.stop}) hold less than one {block_ns} ns block.")
    return FrameLayout(
        n_frames=n_frames,
        dt_ps=dt_ps,
        analysed=analysed,
        blocks=blocks,
        block_times_ns=[time_span_ns(u, frames, dt_ps) for frames in blocks],
        trajectory_start_ns=frame_time_ns(u, 0),
        trajectory_end_ns=frame_time_ns(u, n_frames - 1),
        analysis_start_ns=frame_time_ns(u, analysed.start),
        analysis_end_ns=frame_time_ns(u, analysed.stop - 1),
    )
