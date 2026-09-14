import numpy as np
import pytest

from physicskit.chaos.quantum.billiards import QuantumBilliard, points_in_billiard
from physicskit.chaos.systems.billiards import (
    BunimovichStadium,
    CircleBilliard,
    RectangleBilliard,
    SinaiBilliard,
)


def test_points_in_billiard_excludes_scatterer_hole():
    """The Sinai billiard's boundary polyline has two components (outer square,
    inner scatterer circle); a correct point-in-domain test must exclude the
    scatterer's interior even though it is "inside" the outer square."""
    billiard = SinaiBilliard(cell_size=2.0, scatterer_radius=0.5)
    points = np.array(
        [
            [0.0, 0.0],  # scatterer center: outside the physical domain
            [0.3, 0.0],  # inside the scatterer: outside the physical domain
            [0.7, 0.0],  # outside the scatterer, inside the cell: inside
            [0.9, 0.9],  # near a corner: inside
            [1.5, 1.5],  # outside the cell entirely: outside
        ]
    )
    expected = np.array([False, False, True, True, False])
    assert np.array_equal(points_in_billiard(billiard, points), expected)


def test_points_in_billiard_circle():
    billiard = CircleBilliard(radius=1.0)
    points = np.array([[0.0, 0.0], [0.5, 0.5], [0.99, 0.0], [1.01, 0.0], [2.0, 2.0]])
    expected = np.array([True, True, True, False, False])
    assert np.array_equal(points_in_billiard(billiard, points), expected)


def test_quantum_billiard_rejects_low_resolution():
    with pytest.raises(ValueError):
        QuantumBilliard(CircleBilliard(radius=1.0), resolution=5)


def test_quantum_billiard_eigenvalues_are_positive_and_ascending():
    qb = QuantumBilliard(RectangleBilliard(width=2.0, height=1.0), resolution=80)
    eigenvalues, eigenfunctions = qb.eigenstates(n_states=6)
    assert eigenvalues.shape == (6,)
    assert np.all(eigenvalues > 0.0)
    assert np.all(np.diff(eigenvalues) >= 0.0)
    assert eigenfunctions.shape == (6, *qb.grid()[0].shape)


def test_quantum_billiard_eigenfunctions_vanish_outside_domain():
    qb = QuantumBilliard(CircleBilliard(radius=1.0), resolution=80)
    _, eigenfunctions = qb.eigenstates(n_states=3)
    inside = qb._inside
    assert np.all(np.isnan(eigenfunctions[:, ~inside]))
    assert np.all(np.isfinite(eigenfunctions[:, inside]))


def test_quantum_billiard_matches_analytic_rectangle_spectrum():
    """The Dirichlet rectangle's eigenvalues are known exactly:
    k^2_mn = pi^2 * (m^2/Lx^2 + n^2/Ly^2). The finite-difference solver should
    match this to a few percent at a moderate grid resolution."""
    lx, ly = 2.0, 1.0
    qb = QuantumBilliard(RectangleBilliard(width=lx, height=ly), resolution=150)
    eigenvalues, _ = qb.eigenstates(n_states=6)

    exact = sorted(np.pi**2 * (m**2 / lx**2 + n**2 / ly**2) for m in range(1, 8) for n in range(1, 8))[:6]

    rel_err = np.abs(eigenvalues - np.array(exact)) / np.array(exact)
    assert np.all(rel_err < 0.05)


def test_quantum_billiard_weyl_law_matches_actual_counts_reasonably():
    """Weyl's law is an asymptotic average, not exact for any individual
    eigenvalue, but the smooth counting function should sit close to the
    actual staircase over a modest range of low-lying states."""
    qb = QuantumBilliard(BunimovichStadium(radius=1.0, straight_length=2.0), resolution=110)
    n_states = 20
    wavenumbers = qb.wavenumbers(n_states)
    actual_counts = np.arange(1, n_states + 1)
    predicted_counts = qb.weyl_counting_function(wavenumbers)
    assert np.mean(np.abs(actual_counts - predicted_counts)) < 3.0


def test_quantum_billiard_wavenumbers_are_sqrt_of_eigenvalues():
    qb = QuantumBilliard(CircleBilliard(radius=1.0), resolution=80)
    eigenvalues, _ = qb.eigenstates(4)
    assert np.allclose(qb.wavenumbers(4), np.sqrt(eigenvalues))


def test_quantum_billiard_area_is_close_to_analytic_rectangle_area():
    qb = QuantumBilliard(RectangleBilliard(width=2.0, height=1.5), resolution=150)
    assert qb.area() == pytest.approx(2.0 * 1.5, rel=0.02)
