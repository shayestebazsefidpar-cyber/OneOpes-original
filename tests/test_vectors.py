"""Tests for the `vectors` module: a basis around an axis, and circles around it."""

import numpy as np
import pytest

from oneopes.vectors import circles_around_axis, perpendicular_basis

AXES = [[0.3, -1.2, 2.5], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, -4.0], [1.0, 1.0, 0.0], [1e-3, 1.0, 1e-9]]


@pytest.mark.parametrize("axis", AXES)
def test_perpendicular_basis_is_right_handed_and_orthonormal(axis):
    unit = np.asarray(axis) / np.linalg.norm(axis)
    u, v = perpendicular_basis(unit)
    frame = np.array([u, v, unit])
    np.testing.assert_allclose(frame @ frame.T, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(np.cross(u, v), unit, atol=1e-12)


def test_circles_are_centred_on_the_axis_with_their_radius():
    origin, axis = np.array([1.0, 2.0, 3.0]), np.array([0.0, 0.6, 0.8])
    positions, radii = np.array([0.0, 1.5, 4.0]), np.array([2.0, 1.0, 0.5])
    points = circles_around_axis(origin, axis, positions, radii, 6)

    assert points.shape == (3, 6, 3)
    offsets = points - origin
    along = offsets @ axis
    across = np.linalg.norm(offsets - along[..., np.newaxis] * axis, axis=-1)
    np.testing.assert_allclose(along, np.repeat(positions[:, np.newaxis], 6, axis=1), atol=1e-12)
    np.testing.assert_allclose(across, np.repeat(radii[:, np.newaxis], 6, axis=1), atol=1e-12)


def test_circle_points_are_evenly_spaced_in_angle():
    n = 8
    circle = circles_around_axis(np.zeros(3), np.array([0.0, 0.0, 1.0]), np.array([1.0]), np.array([2.0]), n)[0]
    chords = np.linalg.norm(circle - np.roll(circle, -1, axis=0), axis=1)
    np.testing.assert_allclose(chords, 2 * 2.0 * np.sin(np.pi / n), atol=1e-12)
