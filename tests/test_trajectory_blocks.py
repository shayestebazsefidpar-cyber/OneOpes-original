"""Trajectory blocks: frame step, block length, complete blocks and the frame layout of an analysis."""
import MDAnalysis as mda
import numpy as np
import pytest
from MDAnalysis.coordinates.memory import MemoryReader

from oneopes.trajectory_blocks import Block, complete_blocks, frame_layout, frame_step_ps, frames_per_block


def universe_with_times(n_frames: int, dt_ps: float) -> mda.Universe:
    u = mda.Universe.empty(1, trajectory=True)
    u.load_new(np.zeros((n_frames, 1, 3), dtype=np.float32), format=MemoryReader, dt=dt_ps)
    return u


def test_frame_step_is_the_spacing_between_frames():
    assert frame_step_ps(universe_with_times(5, 20.0)) == 20.0


def test_frame_step_needs_two_frames():
    with pytest.raises(ValueError, match="two frames"):
        frame_step_ps(universe_with_times(1, 20.0))


def test_frames_per_block():
    assert frames_per_block(5.0, 20.0) == 250


@pytest.mark.parametrize("block_ns", [0.0, -1.0, 0.03, float("nan")])
def test_frames_per_block_rejects_bad_lengths(block_ns):
    """Non-positive, NaN, and 0.03 ns, which is 1.5 frames of 20 ps."""
    with pytest.raises(ValueError):
        frames_per_block(block_ns, 20.0)


def test_complete_blocks_skip_the_leftover_frames():
    assert complete_blocks(1, 12, 3) == [range(1, 4), range(4, 7), range(7, 10)]


def test_no_complete_block():
    assert complete_blocks(0, 2, 3) == []


def test_frame_layout_of_a_frame_range():
    """Frames 1-10 of 12 (1 ns apart) in 3 ns blocks: blocks 1-3, 4-6, 7-9, and frame 10 left over."""
    layout = frame_layout(universe_with_times(12, 1000.0), block_ns=3.0, first_frame=1, last_frame=11)
    assert layout.blocks == [Block(range(1, 4), 1.0, 4.0), Block(range(4, 7), 4.0, 7.0), Block(range(7, 10), 7.0, 10.0)]
    assert (layout.trajectory_start_ns, layout.trajectory_end_ns) == (0.0, 11.0)
    assert (layout.analysis_start_ns, layout.analysis_end_ns, layout.analysis_time_ns) == (1.0, 10.0, 9.0)
    assert layout.tail_frames == 1


@pytest.mark.parametrize("first_frame, last_frame, message", [
    (5, 5, "Invalid frame range"),
    (0, 20, "Invalid frame range"),
    (0, 2, "less than one"),
])
def test_frame_layout_rejects_bad_ranges(first_frame, last_frame, message):
    with pytest.raises(ValueError, match=message):
        frame_layout(universe_with_times(12, 1000.0), block_ns=3.0, first_frame=first_frame, last_frame=last_frame)
