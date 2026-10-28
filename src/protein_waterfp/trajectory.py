"""
Per-frame structural pass (Contact, Bridge).

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
