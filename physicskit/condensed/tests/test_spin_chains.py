"""Tests for physicskit.condensed.spin_chains: exact diagonalization checked
against the Bethe ansatz, the free-fermion XX and transverse-field Ising
solutions, CFT finite-size scaling, and area-law vs. logarithmic
entanglement."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.condensed.spin_chains import (
    bethe_ansatz_xxx_ground_energy,
    calabrese_cardy_entropy,
    embed_state,
    energy_gap,
    entanglement_entropy,
    entanglement_profile,
    fit_central_charge,
    lowest_eigenstates,
    spin_chain_basis,
    tfim_free_fermion_spectrum,
    tfim_ground_state_energy,
    tfim_hamiltonian,
    xxz_hamiltonian,
)


def _dense_spin_ops(N):
    # local basis (down, up), matching bit 1 = spin up
    sx = np.array([[0, 1], [1, 0]]) / 2
    sy = np.array([[0, 1j], [-1j, 0]]) / 2
    sz = np.array([[-1, 0], [0, 1]]) / 2

    def site(o, i):
        m = np.array([[1.0]])
        for j in range(N):
            m = np.kron(m, o if j == i else np.eye(2))
        return m

    return [[site(o, i) for i in range(N)] for o in (sx, sy, sz)]


def test_xxz_matches_dense_kronecker_construction():
    N, J, delta, h = 5, 0.8, 0.6, 0.3
    Sx, Sy, Sz = _dense_spin_ops(N)
    H_ref = sum(J * (Sx[i] @ Sx[(i + 1) % N] + Sy[i] @ Sy[(i + 1) % N] + delta * Sz[i] @ Sz[(i + 1) % N]) for i in range(N))
    H_ref = H_ref - h * sum(Sz)
    H, basis = xxz_hamiltonian(N, J=J, delta=delta, h=h)
    np.testing.assert_array_equal(basis, np.arange(2**N))
    np.testing.assert_allclose(H.toarray(), H_ref.real, atol=1e-14)


def test_sz_sectors_partition_the_full_spectrum():
    N = 8
    full = np.linalg.eigvalsh(xxz_hamiltonian(N, delta=0.7, h=0.2)[0].toarray())
    sectors = np.concatenate([np.linalg.eigvalsh(xxz_hamiltonian(N, delta=0.7, h=0.2, n_up=m)[0].toarray()) for m in range(N + 1)])
    np.testing.assert_allclose(np.sort(sectors), full, atol=1e-12)
    assert sum(len(spin_chain_basis(N, n_up=m)) for m in range(N + 1)) == 2**N


@pytest.mark.parametrize("N", [4, 6, 8, 10, 12, 14, 16])
def test_heisenberg_ground_state_matches_bethe_ansatz(N):
    H, _ = xxz_hamiltonian(N, J=1.0, delta=1.0, n_up=N // 2)
    E0, _ = lowest_eigenstates(H, k=1)
    assert E0[0] == pytest.approx(bethe_ansatz_xxx_ground_energy(N), abs=1e-9)


def test_bethe_ansatz_approaches_hulthen_limit():
    N = 2000
    e = bethe_ansatz_xxx_ground_energy(N) / N
    # leading finite-size correction is -pi^2/(12 N^2) (Affleck 1986 via CFT, c=1, v=pi/2)
    assert e == pytest.approx(0.25 - np.log(2), abs=1e-6)


def test_bethe_ansatz_rejects_odd_chain():
    with pytest.raises(ValueError):
        bethe_ansatz_xxx_ground_energy(7)


def test_ferromagnetic_state_energy():
    N, J, delta, h = 7, -1.0, 1.5, 0.4
    H, basis = xxz_hamiltonian(N, J=J, delta=delta, h=h, n_up=N)
    assert basis.tolist() == [2**N - 1]
    assert H.toarray()[0, 0] == pytest.approx(J * delta * N / 4 - h * N / 2)


def test_open_xx_chain_is_free_fermions():
    # H = J sum (SxSx + SySy) = (J/2) sum (c^dag c + h.c.) under Jordan-Wigner;
    # open chain: eps_m = J cos(pi m / (N+1)); fill all negative levels.
    N, J = 10, 1.0
    eps = J * np.cos(np.pi * np.arange(1, N + 1) / (N + 1))
    E_exact = eps[eps < 0].sum()
    H, _ = xxz_hamiltonian(N, J=J, delta=0.0, periodic=False)
    E0, _ = lowest_eigenstates(H, k=1)
    assert E0[0] == pytest.approx(E_exact, abs=1e-10)


@pytest.mark.parametrize("N", [5, 6, 9])
@pytest.mark.parametrize("h", [0.3, 1.0, 2.2])
def test_tfim_full_spectrum_is_free_fermion(N, h):
    H, _ = tfim_hamiltonian(N, J=1.0, h=h)
    np.testing.assert_allclose(tfim_free_fermion_spectrum(N, 1.0, h), np.linalg.eigvalsh(H.toarray()), atol=1e-10)


@pytest.mark.parametrize("parity", [+1, -1])
def test_tfim_parity_sector_spectra(parity):
    N, h = 8, 0.7
    H, basis = tfim_hamiltonian(N, J=1.0, h=h, parity=parity)
    assert len(basis) == 2 ** (N - 1)
    np.testing.assert_allclose(tfim_free_fermion_spectrum(N, 1.0, h, parity=parity), np.linalg.eigvalsh(H.toarray()), atol=1e-10)


@pytest.mark.parametrize("N, h", [(14, 0.5), (14, 1.0), (15, 1.7)])
def test_tfim_lanczos_ground_state_energy(N, h):
    H, _ = tfim_hamiltonian(N, J=1.0, h=h, parity=+1)
    E0, _ = lowest_eigenstates(H, k=1)
    assert E0[0] == pytest.approx(tfim_ground_state_energy(N, 1.0, h), abs=1e-9)


def test_tfim_paramagnetic_gap_is_two_h_minus_j():
    # single-quasiparticle gap eps_{k=0} = 2|h - J|; finite-size corrections
    # ~ exp(-N ln(h/J)) ~ 2e-5 at N=12
    H, _ = tfim_hamiltonian(12, J=1.0, h=2.5)
    assert energy_gap(H) == pytest.approx(2 * (2.5 - 1.0), abs=1e-4)


def test_tfim_ordered_phase_quasi_degenerate():
    H, _ = tfim_hamiltonian(12, J=1.0, h=0.3)
    assert energy_gap(H) < 1e-6


def test_critical_tfim_gap_follows_cft_scaling():
    # E_sigma - E_0 = 2 pi v x_sigma / N with v = 2J and x_sigma = 1/8 (Cardy 1984)
    N = 14
    H, _ = tfim_hamiltonian(N, J=1.0, h=1.0)
    assert N * energy_gap(H) == pytest.approx(np.pi / 2, rel=2e-3)


def test_singlet_entanglement_and_embedding():
    H, basis = xxz_hamiltonian(2, periodic=False, n_up=1)
    _, V = lowest_eigenstates(H, k=1)
    assert entanglement_entropy(V[:, 0], 2, 1, basis=basis) == pytest.approx(np.log(2))
    assert np.linalg.norm(embed_state(V[:, 0], basis, 2)) == pytest.approx(1.0)


def test_critical_ising_entanglement_is_logarithmic_with_c_one_half():
    N = 16
    H, basis = tfim_hamiltonian(N, J=1.0, h=1.0, parity=+1)
    _, V = lowest_eigenstates(H, k=1)
    profile = entanglement_profile(V[:, 0], N, basis=basis)
    c, const = fit_central_charge(profile, N, trim=2)
    assert c == pytest.approx(0.5, abs=0.02)
    ell = np.arange(1, N)
    np.testing.assert_allclose(profile[2:-2], calabrese_cardy_entropy(ell, N, c, const)[2:-2], atol=5e-3)


def test_gapped_ising_entanglement_obeys_area_law():
    N = 16
    H, basis = tfim_hamiltonian(N, J=1.0, h=3.0, parity=+1)
    _, V = lowest_eigenstates(H, k=1)
    profile = entanglement_profile(V[:, 0], N, basis=basis)
    c, _ = fit_central_charge(profile, N, trim=2)
    assert abs(c) < 0.02
    assert profile[N // 2 - 1] - profile[2] < 1e-3


def test_heisenberg_entanglement_has_c_near_one():
    N = 16
    H, basis = xxz_hamiltonian(N, n_up=N // 2)
    _, V = lowest_eigenstates(H, k=1)
    c, _ = fit_central_charge(entanglement_profile(V[:, 0], N, basis=basis), N, trim=2)
    # marginally irrelevant operator gives slow log corrections; c -> 1 only slowly
    assert c == pytest.approx(1.0, abs=0.15)
