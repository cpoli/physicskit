"""Biot-Savart fields against closed-form magnetostatics."""

import numpy as np
import pytest

from physicskit.fields.electrodynamics import MU0
from physicskit.fields.magnetostatics import (
    biot_savart_field,
    circular_loop_path,
    infinite_wire_field,
    loop_axial_field,
    magnetic_dipole_field,
    solenoid_axial_field,
    solenoid_path,
)


def test_current_loop_on_axis_field():
    """B_z = mu0 I R^2 / (2 (R^2 + z^2)^{3/2}) on the axis (Griffiths Example 5.6)."""
    I, R = 2.5, 0.2
    z = np.linspace(-1.0, 1.0, 21)
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    B = biot_savart_field(circular_loop_path(R, n_segments=1024), pts, current=I)
    assert B[:, 2] == pytest.approx(loop_axial_field(I, R, z), rel=1e-5)
    assert np.max(np.abs(B[:, :2])) < 1e-10 * np.max(np.abs(B[:, 2]))


def test_polygon_loop_error_is_second_order():
    """A regular n-gon's center field converges to mu0 I / 2R as 1/n^2."""
    R = 1.0
    exact = loop_axial_field(1.0, R, 0.0)
    err = [abs(biot_savart_field(circular_loop_path(R, n), [[0, 0, 0]])[0, 2] / exact - 1) for n in (64, 128)]
    assert err[0] / err[1] == pytest.approx(4.0, rel=0.01)


def test_loop_far_field_is_magnetic_dipole():
    I, R = 1.0, 0.05
    m = np.array([0.0, 0.0, I * np.pi * R**2])
    pts = 3.0 * np.array([[1.0, 0.0, 0.0], [0.0, 0.6, 0.8], [0.3, -0.4, np.sqrt(0.75)]])
    B = biot_savart_field(circular_loop_path(R, 2048), pts, current=I)
    assert B == pytest.approx(magnetic_dipole_field(m, pts), rel=1e-3, abs=1e-6 * np.max(np.abs(B)))


def test_finite_solenoid_axial_field():
    """A tightly wound helix matches the current-sheet result
    (mu0 n I / 2)[cos theta_2 - cos theta_1] along the axis (Griffiths Problem 5.11)."""
    I, R, L, N = 1.0, 0.02, 0.2, 200
    path = solenoid_path(R, L, N, points_per_turn=64)
    z = np.array([0.0, 0.05, 0.1, 0.15])
    pts = np.column_stack([np.zeros_like(z), np.zeros_like(z), z])
    B = biot_savart_field(path, pts, current=I)
    assert B[:, 2] == pytest.approx(solenoid_axial_field(I, N, R, L, z), rel=2e-3)
    # deep inside: mu0 n I, and half that at the end
    assert B[0, 2] == pytest.approx(MU0 * N / L * I, rel=0.02)
    assert B[2, 2] == pytest.approx(0.5 * MU0 * N / L * I, rel=0.02)


def test_straight_wire_and_amperes_law():
    """|B| = mu0 I / (2 pi s) around a long wire, and its circulation is mu0 I."""
    I = 3.0
    wire = np.array([[0.0, 0.0, -1e3], [0.0, 0.0, 1e3]])
    s = 0.25
    phi = np.linspace(0.0, 2 * np.pi, 400, endpoint=False)
    pts = np.column_stack([s * np.cos(phi), s * np.sin(phi), np.zeros_like(phi)])
    B = biot_savart_field(wire, pts, current=I)
    tangent = np.column_stack([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    B_phi = np.sum(B * tangent, axis=1)
    assert B_phi == pytest.approx(infinite_wire_field(I, s), rel=1e-9)
    circulation = np.sum(B_phi) * s * (2 * np.pi / len(phi))
    assert circulation == pytest.approx(MU0 * I, rel=1e-9)


def test_field_is_divergence_free():
    path = circular_loop_path(0.3, 256)
    x0 = np.array([0.2, 0.1, 0.15])
    h = 1e-5
    div = 0.0
    for k in range(3):
        e = np.zeros(3)
        e[k] = h
        div += (biot_savart_field(path, [x0 + e])[0, k] - biot_savart_field(path, [x0 - e])[0, k]) / (2 * h)
    scale = np.linalg.norm(biot_savart_field(path, [x0])[0]) / 0.3
    assert abs(div) < 1e-6 * scale


def test_points_array_shape_is_preserved_and_bad_path_raises():
    B = biot_savart_field(circular_loop_path(1.0, 16), np.zeros((4, 5, 3)) + [0, 0, 2.0])
    assert B.shape == (4, 5, 3)
    with pytest.raises(ValueError):
        biot_savart_field(np.zeros((1, 3)), [[0, 0, 1]])
