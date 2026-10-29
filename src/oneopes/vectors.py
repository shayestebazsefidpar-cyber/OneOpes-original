"""Vector geometry around an axis; lengths in any one unit."""

import numpy as np
from scipy.linalg import null_space


def perpendicular_basis(axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Unit vectors u, v perpendicular to the unit vector ``axis``, with (u, v, axis) right-handed."""
    u = null_space(axis[np.newaxis])[:, 0]
    return u, np.cross(axis, u)


def circles_around_axis(origin: np.ndarray, axis: np.ndarray, positions: np.ndarray, radii: np.ndarray,
                        n_points: int) -> np.ndarray:
    """``n_points`` on each circle around ``axis``, centred ``positions[i]`` from ``origin`` with radius ``radii[i]``.

    Shape (len(positions), n_points, 3).
    """
    u, v = perpendicular_basis(axis)
    angles = np.linspace(0.0, 2 * np.pi, n_points, endpoint=False)
    directions = np.outer(np.cos(angles), u) + np.outer(np.sin(angles), v)
    centres = origin + np.outer(positions, axis)
    return centres[:, np.newaxis] + radii[:, np.newaxis, np.newaxis] * directions
