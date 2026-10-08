"""
Block plan and the per-frame structural pass (Contact, Bridge).

Blocks: consecutive, non-overlapping groups of ``round(block_ns / dt)``
frames starting at ``first_frame``; only complete blocks are analysed and the
leftover frames are reported, never silently used.

Per frame (distances in A, minimum image):
* Contact: candidate's minimum distance to any ligand heavy atom <= 5 A.
* Bridge: at least one water oxygen within 3.5 A of the candidate AND within
  3.5 A of some ligand heavy atom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from MDAnalysis.lib.distances import distance_array


CONTACT_CUTOFF_A = 5.0
BRIDGE_WATER_CUTOFF_A = 3.5
BLOCK_NS_DEFAULT = 5.0


@dataclass(frozen=True)
class BlockPlan:
    dt_ps: float
    frames_per_block: int
    first_frame: int
    last_frame_exclusive: int
    """End (exclusive) of the frame range considered (before trimming to whole blocks)."""
    blocks: tuple[tuple[int, int], ...]
    """(start, end_exclusive) frame index of each complete block."""

    @property
    def analysed_frames(self) -> tuple[int, int]:
        return (self.blocks[0][0], self.blocks[-1][1]) if self.blocks else (self.first_frame, self.first_frame)

    @property
    def unused_tail_frames(self) -> int:
        return self.last_frame_exclusive - self.analysed_frames[1]


def plan_blocks(n_frames: int, dt_ps: float, block_ns: float = BLOCK_NS_DEFAULT,
                first_frame: int = 0, last_frame: int | None = None) -> BlockPlan:
    """Complete ``block_ns`` blocks within frames [first_frame, last_frame)."""
    if not (dt_ps > 0 and np.isfinite(dt_ps)):
        raise ValueError(f"Trajectory time step must be > 0, got {dt_ps}.")
    if not (block_ns > 0 and np.isfinite(block_ns)):
        raise ValueError(f"block_ns must be > 0, got {block_ns}.")
    last = n_frames if last_frame is None else last_frame
    if not (0 <= first_frame < last <= n_frames):
        raise ValueError(f"Invalid frame range [{first_frame}, {last}) for a {n_frames}-frame trajectory.")
    fpb = int(round(block_ns * 1000.0 / dt_ps))
    if fpb < 1 or not np.isclose(fpb * dt_ps, block_ns * 1000.0, rtol=0, atol=1e-6 * max(1.0, block_ns * 1000.0)):
        raise ValueError(f"block_ns={block_ns} is not a whole number of {dt_ps} ps frames.")
    n_blocks = (last - first_frame) // fpb
    if n_blocks < 1:
        raise ValueError(f"Frames [{first_frame}, {last}) hold less than one {block_ns} ns block.")
    blocks = tuple((first_frame + b * fpb, first_frame + (b + 1) * fpb) for b in range(n_blocks))
    return BlockPlan(dt_ps=float(dt_ps), frames_per_block=fpb, first_frame=first_frame,
                     last_frame_exclusive=last, blocks=blocks)


def trajectory_dt_ps(u) -> float:
    """Constant frame spacing (ps); fails if the spacing is not uniform."""
    n = len(u.trajectory)
    if n < 2:
        raise ValueError("The trajectory needs at least two frames.")
    t0 = u.trajectory[0].time
    dt = u.trajectory[1].time - t0
    t_last = u.trajectory[n - 1].time
    if not np.isclose(t_last - t0, dt * (n - 1), rtol=1e-6, atol=1e-3):
        raise ValueError("Trajectory frames are not evenly spaced in time.")
    return float(dt)


@dataclass
class StructuralBlock:
    contact_occupancy: np.ndarray
    bridge_occupancy: np.ndarray
    n_frames: int


def structural_block(u, start: int, end: int, cand_idx, ligand_heavy_idx, water_o_idx) -> StructuralBlock:
    """Contact / Bridge occupancies of each candidate over frames [start, end)."""
    cand_idx = np.asarray(cand_idx)
    lig_idx = np.asarray(ligand_heavy_idx)
    contact = np.zeros(len(cand_idx))
    bridge = np.zeros(len(cand_idx))
    for fi in range(start, end):
        ts = u.trajectory[fi]
        box = ts.dimensions
        pos = u.atoms.positions
        cand_pos, lig_pos, water_pos = pos[cand_idx], pos[lig_idx], pos[water_o_idx]
        contact += distance_array(cand_pos, lig_pos, box=box).min(axis=1) <= CONTACT_CUTOFF_A
        water_near_lig = distance_array(water_pos, lig_pos, box=box).min(axis=1) <= BRIDGE_WATER_CUTOFF_A
        near_cand = distance_array(cand_pos, water_pos[water_near_lig], box=box) <= BRIDGE_WATER_CUTOFF_A
        bridge += near_cand.any(axis=1) if near_cand.size else 0
    n = end - start
    return StructuralBlock(contact / n, bridge / n, n)
