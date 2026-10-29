"""Trajectory blocks: consecutive runs of the same number of frames, each a :class:`Block` of frames and times."""
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import xarray as xr
from MDAnalysis import Universe

from oneopes.errors import InputError

BLOCK_NS_DEFAULT = 5.0


def frame_step_ps(u: Universe, n_frames: int | None = None) -> float:
    """Time between frames (ps), checked over the first ``n_frames`` frames (default: all);
    raises InputError unless they are evenly spaced."""
    n = len(u.trajectory) if n_frames is None else n_frames
    if n < 2:
        raise InputError("The trajectory needs at least two frames.")
    t0 = u.trajectory[0].time
    dt = u.trajectory[1].time - t0
    if not np.isclose(u.trajectory[n - 1].time - t0, dt * (n - 1), rtol=1e-6, atol=1e-3):
        raise InputError("Trajectory frames are not evenly spaced in time.")
    if not (dt > 0 and np.isfinite(dt)):
        raise InputError(f"Trajectory time step must be > 0, got {dt}.")
    return float(dt)


def frames_per_block(block_ns: float, dt_ps: float) -> int:
    """Frames in a ``block_ns`` block; raises InputError unless that is a whole, positive number of frames."""
    n = round(block_ns * 1000.0 / dt_ps)
    if n < 1 or not np.isclose(n * dt_ps, block_ns * 1000.0, rtol=0, atol=1e-6 * max(1.0, block_ns * 1000.0)):
        raise InputError(f"block_ns={block_ns} is not a whole number of {dt_ps} ps frames.")
    return n


def frame_time_ns(u: Universe, frame: int) -> float:
    """Simulation time of a frame (ns)."""
    return float(u.trajectory[frame].time) / 1000.0


def complete_blocks(start: int, stop: int, length: int) -> list[range]:
    """The complete blocks of ``length`` frames in frames [start, stop); leftover frames at the end are skipped."""
    return [range(first, first + length) for first in range(start, stop - length + 1, length)]


@dataclass(frozen=True)
class Block:
    """One trajectory block: its frames and its start and end time (ns)."""
    frames: range
    start_ns: float
    end_ns: float


def block_of(u: Universe, frames: range, dt_ps: float) -> Block:
    """The block of ``frames``: it starts at the time of its first frame and lasts ``len(frames)`` frames."""
    start_ns = frame_time_ns(u, frames.start)
    return Block(frames=frames, start_ns=start_ns, end_ns=start_ns + len(frames) * dt_ps / 1000.0)


def label_block(measurement: xr.Dataset, block: Block) -> xr.Dataset:
    """A block's measurement, labelled with the block's start and end time (ns) and frame count."""
    return measurement.assign_coords(t_start_ns=block.start_ns, t_end_ns=block.end_ns, n_frames=len(block.frames))


def stack_blocks(blocks: Sequence[xr.Dataset]) -> xr.Dataset:
    """Block measurements along a ``block`` dimension numbered from 0, with each block's mid time ``t_mid_ns``."""
    stacked = xr.concat(blocks, dim="block", data_vars="all", coords=["t_start_ns", "t_end_ns", "n_frames"],
                        join="exact")
    stacked = stacked.assign_coords(block=np.arange(len(blocks)))
    return stacked.assign_coords(t_mid_ns=(stacked.t_start_ns + stacked.t_end_ns) / 2)


@dataclass(frozen=True)
class FrameLayout:
    """The analysed frames of a trajectory, the complete blocks in them, and their times (ns)."""
    n_frames: int
    """Frames in the whole trajectory."""
    dt_ps: float
    analysed: range
    blocks: list[Block]
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
        return self.analysed.stop - self.blocks[-1].frames.stop

    def describe(self) -> str:
        """The layout in one line, for the log."""
        return (f"{self.n_frames} frames of {self.dt_ps} ps "
                f"({self.trajectory_start_ns:g}-{self.trajectory_end_ns:g} ns); "
                f"analysed frames {self.analysed.start}-{self.analysed.stop - 1} ({self.analysis_time_ns:g} ns) "
                f"in {len(self.blocks)} blocks of {len(self.blocks[0].frames)} frames, {self.tail_frames} left over")


@dataclass(frozen=True)
class FrameWindow:
    """The frames to analyse; also command options."""
    first_frame: int = 0
    """First analysed frame."""
    last_frame: int | None = None
    """Exclusive end frame (default: end of the trajectory)."""

    def frames(self, n_frames: int) -> range:
        """The window in a trajectory of ``n_frames`` frames; raises InputError unless it is a non-empty part of it."""
        frames = range(self.first_frame, n_frames if self.last_frame is None else self.last_frame)
        if not 0 <= frames.start < frames.stop <= n_frames:
            raise InputError(f"Invalid frame range [{frames.start}, {frames.stop}) for a {n_frames}-frame trajectory.")
        return frames


def frame_layout(u: Universe, block_ns: float, window: FrameWindow = FrameWindow()) -> FrameLayout:
    """Split the ``window`` frames of ``u`` (default: all) into complete ``block_ns`` blocks.

    Raises InputError for an invalid frame window, unevenly spaced frames, or fewer frames than one block.
    """
    n_frames = len(u.trajectory)
    analysed = window.frames(n_frames)
    dt_ps = frame_step_ps(u)
    block_frames = complete_blocks(analysed.start, analysed.stop, frames_per_block(block_ns, dt_ps))
    if not block_frames:
        raise InputError(f"Frames [{analysed.start}, {analysed.stop}) hold less than one {block_ns} ns block.")
    return FrameLayout(
        n_frames=n_frames,
        dt_ps=dt_ps,
        analysed=analysed,
        blocks=[block_of(u, frames, dt_ps) for frames in block_frames],
        trajectory_start_ns=frame_time_ns(u, 0),
        trajectory_end_ns=frame_time_ns(u, n_frames - 1),
        analysis_start_ns=frame_time_ns(u, analysed.start),
        analysis_end_ns=frame_time_ns(u, analysed.stop - 1),
    )
