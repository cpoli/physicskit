"""Spot-check each chapter against known analytic results, so regressions in
the underlying physics (not just raw unitarity) get caught."""

import numpy as np

from physicskit.quantum.chapters.entanglement import AharonovBohmRing, BellCorrelations
from physicskit.quantum.chapters.harmonic_spin import HarmonicOscillator, ThermalState
from physicskit.quantum.chapters.hydrogen_am import HydrogenOrbital
from physicskit.quantum.chapters.potentials import FiniteSquareWell, StadiumBilliard2D
from physicskit.quantum.chapters.wave_packets import QuantumRevival
from physicskit.quantum.core.eigensolvers import NumerovSolver, harmonic_well, infinite_well
from physicskit.quantum.visualizers.phase_space import wigner_transform


def test_numerov_matches_infinite_well_and_harmonic_oscillator():
    x = np.linspace(0, 1, 400)
    energies = NumerovSolver(x, infinite_well()).solve(5).energies
    analytic = (np.arange(1, 6) ** 2 * np.pi**2) / 2
    assert np.allclose(energies, analytic, rtol=1e-3)

    x2 = np.linspace(-10, 10, 800)
    energies2 = NumerovSolver(x2, harmonic_well()).solve(5).energies
    assert np.allclose(energies2, np.arange(5) + 0.5, atol=1e-4)


def test_coherent_state_tracks_classical_trajectory():
    ho = HarmonicOscillator()
    x = np.linspace(-15, 15, 3000)
    alpha = 3 + 1j
    for t in (0.0, 1.3, 4.0):
        psi_t = ho.coherent_wavefunction(alpha, x, t=t)
        x_mean = np.real(np.trapezoid(x * np.abs(psi_t) ** 2, x))
        x_classical, _ = ho.coherent_trajectory(alpha, np.array([t]))
        assert abs(x_mean - x_classical[0]) < 1e-6


def test_thermal_state_broadens_with_temperature():
    ho = HarmonicOscillator()
    var_cold = ThermalState(ho, T=0.1).variance()
    var_hot = ThermalState(ho, T=5.0).variance()
    assert var_hot > var_cold


def test_hydrogen_orbitals_normalized_with_correct_energy():
    for n, l, m in [(1, 0, 0), (2, 1, 0), (3, 2, 1)]:
        orb = HydrogenOrbital(n, l, m)
        assert abs(orb.check_normalization() - 1.0) < 1e-6
        assert abs(orb.energy - (-1 / (2 * n**2))) < 1e-12


def test_quantum_revival_reassembles_at_revival_time():
    qr = QuantumRevival(L=1.0, n_max=250)
    x = np.linspace(0, qr.L, 1500)
    psi0_func = qr.gaussian_initial_state(x0=0.3, sigma=0.03)
    psi0 = psi0_func(x)
    c = qr.eigenbasis_coefficients(psi0_func)

    fidelity_scrambled = qr.fidelity_to_initial(x, 0.1 * qr.revival_time, c, psi0)
    fidelity_revived = qr.fidelity_to_initial(x, qr.revival_time, c, psi0)

    assert fidelity_revived > 0.999
    assert fidelity_scrambled < 0.5


def test_wigner_transform_matches_ground_state_analytic_form():
    ho = HarmonicOscillator()
    x = np.linspace(-8, 8, 250)
    psi0 = ho.eigenfunction(0, x)
    xg, p, W = wigner_transform(x, psi0, hbar=1.0, n_p=120)
    X, P = np.meshgrid(x, p, indexing="ij")
    analytic = (1 / np.pi) * np.exp(-(X**2) - P**2)
    assert np.max(np.abs(W - analytic)) < 1e-10


def test_bell_singlet_violates_chsh_bound():
    bc = BellCorrelations()
    assert abs(bc.chsh_optimal() - 2 * np.sqrt(2)) < 1e-10
    assert bc.chsh_optimal() > 2.0  # violates the classical (local hidden-variable) bound


def test_aharonov_bohm_energy_periodic_in_flux_quantum():
    ring = AharonovBohmRing(R=1.0)
    Phi0 = ring.flux_quantum
    n = np.arange(-3, 4)
    e0 = ring.energy(n, 0.0)
    e_shifted = ring.energy(n + 1, Phi0)  # shifting both n and Phi by one quantum is a relabeling
    assert np.allclose(e0, e_shifted)
    assert abs(ring.aharonov_bohm_phase(Phi0) - 2 * np.pi) < 1e-12


def test_barrier_scattering_matches_sub_barrier_tunneling_formula():
    # A "well" with negative V0 is a barrier of height |V0|; below the
    # barrier (E < |V0|) this is the classically forbidden, sub-barrier
    # tunneling regime, where k2 is purely imaginary.
    E, Vb, a = 4.5, 6.0, 1.0
    barrier = FiniteSquareWell(V0=-Vb, width=a)
    result = barrier.scattering(E)

    kappa = np.sqrt(2 * (Vb - E))
    T_expected = 1 / (1 + Vb**2 * np.sinh(kappa * a) ** 2 / (4 * E * (Vb - E)))
    assert abs(result.T - T_expected) < 1e-12
    assert abs(result.k2.real) < 1e-12  # purely imaginary in the forbidden region
    assert 0 < result.T < 1

    # Above the barrier (E > |V0|), T should match the ordinary real-k2 formula.
    E_above = 10.0
    result_above = barrier.scattering(E_above)
    k2_expected = np.sqrt(2 * (E_above - Vb))
    T_expected_above = 1 / (1 + Vb**2 * np.sin(k2_expected * a) ** 2 / (4 * E_above * (E_above - Vb)))
    assert abs(result_above.T - T_expected_above) < 1e-10
    assert abs(result_above.k2.imag) < 1e-12


def test_stadium_billiard_grid_is_isotropic_and_states_are_normalized():
    # A mismatched dx/dy staircases the semicircular caps as ellipses
    # rather than circles, giving resolution-dependent, non-monotonically
    # converging eigenvalues -- grid() must keep dx == dy.
    sb = StadiumBilliard2D(L=1.0, R=0.5)
    for n_points in (50, 90, 150):
        (X, Y), x, y = sb.grid(n_points)
        dx, dy = x[1] - x[0], y[1] - y[0]
        # dy matches dx up to the rounding of n_y to an integer point count
        assert abs(dx - dy) / dx < 0.02

    energies, wavefunctions, X, Y, mask = sb.solve(n_points=90, n_states=4)
    assert np.all(np.diff(energies) > 0)
    assert np.all(energies > 0)
    dx, dy = X[1, 0] - X[0, 0], Y[0, 1] - Y[0, 0]
    for psi in wavefunctions:
        assert abs(np.sum(psi**2) * dx * dy - 1.0) < 1e-6
        assert np.all(psi[~mask] == 0.0)
