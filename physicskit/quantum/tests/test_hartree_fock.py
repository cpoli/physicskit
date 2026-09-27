"""Tests for physicskit.quantum.chapters.hartree_fock against the textbook
results of Szabo and Ostlund (H2, HeH+, He in STO-3G), the Hartree-Fock
limit of helium, and exact Gaussian-integral identities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.integrate import quad

from physicskit.quantum.chapters.hartree_fock import (
    GaussianS,
    electron_repulsion_tensor,
    even_tempered_s_basis,
    kinetic_matrix,
    nuclear_attraction_matrix,
    overlap_matrix,
    restricted_hartree_fock,
    sto3g_1s,
)

ORIGIN = np.zeros(3)


def _h2(R=1.4):
    pos = np.array([[0.0, 0.0, 0.0], [R, 0.0, 0.0]])
    return [sto3g_1s(1.24, p) for p in pos], pos


def test_single_gaussian_integrals_closed_form():
    a = 0.8
    g = [GaussianS([a], [1.0], ORIGIN)]
    assert overlap_matrix(g)[0, 0] == pytest.approx(1.0)
    assert kinetic_matrix(g)[0, 0] == pytest.approx(1.5 * a)
    # <1/r> = 2 sqrt(2a/pi) for a normalized Gaussian
    assert nuclear_attraction_matrix(g, [1.0], [ORIGIN])[0, 0] == pytest.approx(-2 * np.sqrt(2 * a / np.pi))
    # (gg|gg) = 2 sqrt(a/pi)
    assert electron_repulsion_tensor(g)[0, 0, 0, 0] == pytest.approx(2 * np.sqrt(a / np.pi))


def test_off_center_nuclear_attraction_matches_quadrature():
    # <g| -1/|r - C| |g> = -erf(sqrt(2a) d)/d for a normalized Gaussian at distance d
    a, d = 0.6, 1.3
    g = [GaussianS([a], [1.0], ORIGIN)]
    V = nuclear_attraction_matrix(g, [1.0], [[0.0, 0.0, d]])[0, 0]
    rho = lambda r: (2 * a / np.pi) ** 1.5 * np.exp(-2 * a * r**2)  # noqa: E731
    shell = lambda r: 4 * np.pi * r**2 * rho(r) / max(r, d)  # noqa: E731
    assert V == pytest.approx(-quad(shell, 0, 20, points=[d])[0], rel=1e-9)


def test_eri_symmetry_and_positive_definiteness():
    basis, _ = _h2()
    basis += [GaussianS([0.3], [1.0], [0.7, 0.4, 0.0])]
    eri = electron_repulsion_tensor(basis)
    for perm in [(1, 0, 2, 3), (0, 1, 3, 2), (2, 3, 0, 1)]:
        np.testing.assert_allclose(eri, eri.transpose(perm), atol=1e-14)
    n = len(basis)
    assert np.linalg.eigvalsh(eri.reshape(n * n, n * n)).min() > -1e-12


def test_h2_sto3g_szabo_ostlund():
    basis, pos = _h2()
    S = overlap_matrix(basis)
    assert S[0, 1] == pytest.approx(0.6593, abs=1e-4)  # Szabo and Ostlund Eq. 3.229
    res = restricted_hartree_fock(basis, [1, 1], pos, 2)
    assert res.converged
    assert res.electronic_energy == pytest.approx(-1.8310, abs=1e-4)
    assert res.energy == pytest.approx(-1.1167, abs=1e-4)
    np.testing.assert_allclose(res.orbital_energies, [-0.5782, 0.6703], atol=1e-4)
    # closed-shell density is idempotent: P S P = 2 P, and tr(PS) = N
    np.testing.assert_allclose(res.density @ S @ res.density, 2 * res.density, atol=1e-10)
    assert np.trace(res.density @ S) == pytest.approx(2.0)


def test_heh_plus_szabo_ostlund():
    # Szabo and Ostlund Appendix B: zeta(He) = 2.0925, zeta(H) = 1.24, R = 1.4632
    pos = np.array([[0.0, 0.0, 0.0], [1.4632, 0.0, 0.0]])
    basis = [sto3g_1s(2.0925, pos[0]), sto3g_1s(1.24, pos[1])]
    res = restricted_hartree_fock(basis, [2, 1], pos, 2)
    assert res.energy == pytest.approx(-2.860662, abs=1e-5)
    assert res.electronic_energy == pytest.approx(-4.227529, abs=1e-5)
    np.testing.assert_allclose(res.orbital_energies, [-1.597448, -0.061670], atol=1e-5)


def test_helium_sto3g_and_hartree_fock_limit():
    he = restricted_hartree_fock([sto3g_1s(1.69, ORIGIN)], [2], [ORIGIN], 2)
    assert he.energy == pytest.approx(-2.8078, abs=1e-4)
    # even-tempered s basis converges variationally to the HF limit -2.8616800 (Clementi and Roetti 1974)
    energies = [restricted_hartree_fock(even_tempered_s_basis(ORIGIN, n, 0.05, 2.5), [2], [ORIGIN], 2).energy for n in (6, 10, 14)]
    assert energies[0] > energies[1] > energies[2] > -2.8616800
    assert energies[-1] == pytest.approx(-2.8616800, abs=2e-5)


def test_helium_virial_theorem():
    basis = even_tempered_s_basis(ORIGIN, 14, 0.05, 2.5)
    res = restricted_hartree_fock(basis, [2], [ORIGIN], 2)
    kinetic = np.sum(res.density * kinetic_matrix(basis))
    assert -res.energy / kinetic == pytest.approx(1.0, abs=1e-4)  # E = -<T> at the HF optimum


def test_h2_bond_is_bound_and_equilibrium_near_1_35_bohr():
    R = np.linspace(1.1, 1.8, 29)
    E = []
    for r in R:
        basis, pos = _h2(r)
        E.append(restricted_hartree_fock(basis, [1, 1], pos, 2).energy)
    E = np.array(E)
    assert R[np.argmin(E)] == pytest.approx(1.346, abs=0.03)  # STO-3G equilibrium ~1.346 bohr
    # one STO-3G hydrogen atom: <phi| -nabla^2/2 - 1/r |phi> = -0.4666
    h = [sto3g_1s(1.24, ORIGIN)]
    E_H = (kinetic_matrix(h) + nuclear_attraction_matrix(h, [1.0], [ORIGIN]))[0, 0]
    assert E_H == pytest.approx(-0.4666, abs=1e-4)
    assert E.min() < 2 * E_H  # the molecule is bound


def test_rejects_odd_electron_count():
    basis, pos = _h2()
    with pytest.raises(ValueError):
        restricted_hartree_fock(basis, [1, 1], pos, 1)
