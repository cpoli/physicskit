"""Tests for physicskit.quantum.chapters.hydrogen_am: the quantum-number
validation, density, and most_probable_radius paths not already
exercised by test_physics_checks.py/test_animations.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum.chapters.hydrogen_am import HydrogenOrbital, radial_wavefunction


def test_radial_wavefunction_rejects_l_greater_or_equal_n():
    with pytest.raises(ValueError):
        radial_wavefunction(n=1, l=1, r=np.array([1.0]))


def test_hydrogen_orbital_rejects_invalid_l():
    with pytest.raises(ValueError):
        HydrogenOrbital(n=1, l=1, m=0)
    with pytest.raises(ValueError):
        HydrogenOrbital(n=2, l=-1, m=0)


def test_hydrogen_orbital_rejects_m_out_of_range():
    with pytest.raises(ValueError):
        HydrogenOrbital(n=2, l=0, m=1)


def test_hydrogen_orbital_density_matches_abs_psi_squared():
    orbital = HydrogenOrbital(n=2, l=1, m=0)
    r = np.array([1.0, 2.0, 3.0])
    theta = np.array([0.3, 1.0, 2.0])
    phi = np.array([0.1, 0.5, 1.5])
    np.testing.assert_allclose(orbital.density(r, theta, phi), np.abs(orbital.psi(r, theta, phi)) ** 2)


def test_1s_orbital_most_probable_radius_equals_bohr_radius():
    # The textbook result: the 1s orbital's most probable radius is exactly a0.
    orbital = HydrogenOrbital(n=1, l=0, m=0, Z=1, a0=1.0)
    assert orbital.most_probable_radius() == pytest.approx(1.0, rel=1e-3)


def test_most_probable_radius_scales_inversely_with_z():
    orbital_z1 = HydrogenOrbital(n=1, l=0, m=0, Z=1, a0=1.0)
    orbital_z2 = HydrogenOrbital(n=1, l=0, m=0, Z=2, a0=1.0)
    assert orbital_z2.most_probable_radius() == pytest.approx(orbital_z1.most_probable_radius() / 2, rel=1e-2)
