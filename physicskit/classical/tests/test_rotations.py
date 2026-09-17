"""Targeted checks for physicskit.classical.systems.rotations beyond energy conservation
(covered in test_conservation.py): the closed-form/harmonic-approximation
precession and nutation frequencies for HeavySymmetricTop, validated
against an FFT of an actual integrated trajectory.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.classical.systems.rotations import (
    EulersDisk,
    HeavySymmetricTop,
    Rattleback,
    eulers_disk_theta_analytic,
    find_theta_equilibrium,
    nutation_frequency,
    precession_frequency,
)


def _measured_nutation_frequency(t: np.ndarray, theta_t: np.ndarray) -> float:
    dt = t[1] - t[0]
    spectrum = np.abs(np.fft.rfft(theta_t - theta_t.mean()))
    freqs = np.fft.rfftfreq(len(theta_t), d=dt) * 2.0 * np.pi
    peak_idx = np.argmax(spectrum[1:]) + 1
    return float(freqs[peak_idx])


def test_theta_equilibrium_is_a_true_minimum_of_effective_potential():
    I1, I3, Mgl = 1.0, 0.5, 9.81
    p_phi, p_psi = 8.78, 10.0
    theta_eq = find_theta_equilibrium(p_phi, p_psi, I1, I3, Mgl)
    assert 0.0 < theta_eq < np.pi

    from physicskit.classical.systems.rotations import effective_potential_symmetric_top

    v_at_eq = effective_potential_symmetric_top(theta_eq, p_phi, p_psi, I1, I3, Mgl)
    for delta in (-0.05, 0.05):
        assert effective_potential_symmetric_top(theta_eq + delta, p_phi, p_psi, I1, I3, Mgl) > v_at_eq


@pytest.mark.slow
def test_nutation_frequency_matches_fft_of_real_trajectory():
    I1, I3, M, l, g = 1.0, 0.5, 1.0, 1.0, 9.81
    top = HeavySymmetricTop([0.0, 0.5, 0.0], [0.0, 0.0, 20.0], I1=I1, I3=I3, M=M, l=l, g=g)
    q0, qdot0 = top.q.copy(), top.qdot.copy()
    p_phi, _, p_psi = top.momentum(q0, qdot0)

    theta_eq = find_theta_equilibrium(p_phi, p_psi, I1, I3, M * g * l)
    predicted = nutation_frequency(p_phi, p_psi, I1, I3, M * g * l, theta_eq=theta_eq)

    result = top.integrate((0, 10.0), dt=2e-5, method="implicit_midpoint")
    measured = _measured_nutation_frequency(result.t, result.q[:, 1])

    assert abs(measured - predicted) / predicted < 0.02


@pytest.mark.slow
def test_precession_frequency_matches_mean_phi_rate():
    I1, I3, M, l, g = 1.0, 0.5, 1.0, 1.0, 9.81
    top = HeavySymmetricTop([0.0, 0.5, 0.0], [0.0, 0.0, 20.0], I1=I1, I3=I3, M=M, l=l, g=g)
    q0, qdot0 = top.q.copy(), top.qdot.copy()
    p_phi, _, p_psi = top.momentum(q0, qdot0)

    theta_eq = find_theta_equilibrium(p_phi, p_psi, I1, I3, M * g * l)
    predicted = precession_frequency(p_phi, p_psi, I1, theta_eq)

    result = top.integrate((0, 10.0), dt=2e-5, method="implicit_midpoint")
    measured = (result.q[-1, 0] - result.q[0, 0]) / (result.t[-1] - result.t[0])

    assert abs(measured - predicted) / abs(predicted) < 0.02


def test_eulers_disk_theta_matches_closed_form_solution():
    """Well below the collapse time (where the regularizing theta_floor
    never engages), the numerically integrated theta(t) should match the
    closed-form solution `eulers_disk_theta_analytic`."""
    theta0, decay_rate = 0.5, 0.02
    t_collapse = theta0**2 / (2.0 * decay_rate)

    disk = EulersDisk(theta0, decay_rate=decay_rate, precession_const=1.0, theta_floor=1e-4)
    t_end = 0.5 * t_collapse
    dt = 1e-3
    result = disk.integrate((0.0, t_end), dt=dt, method="rk4")

    predicted = eulers_disk_theta_analytic(result.t, theta0, decay_rate)
    np.testing.assert_allclose(result.y[:, 0], predicted, atol=1e-3)


def test_eulers_disk_precession_speeds_up_as_it_runs_down():
    """The whole point of the model: phi advances faster and faster per
    unit time as theta shrinks, since dphi/dt = precession_const/theta."""
    disk = EulersDisk(0.5, decay_rate=0.02, precession_const=1.0, theta_floor=1e-4)
    result = disk.integrate((0.0, 5.0), dt=1e-3, method="rk4")
    phi_rate = np.diff(result.y[:, 1]) / np.diff(result.t)
    # Later phi-rate (disk flatter, closer to collapse) must exceed the
    # earlier phi-rate (disk still steep).
    assert phi_rate[-1] > phi_rate[len(phi_rate) // 10]


def test_rattleback_reverses_spin_direction():
    """Starting spun up hard about n3 with only a tiny rocking
    perturbation, the toy model should show n3 collapse through zero and
    become substantially negative (a genuine reversal), not merely decay
    toward zero and stay positive."""
    system = Rattleback([0.01, 0.01, 3.0])
    result = system.integrate((0.0, 20.0), dt=5e-3, method="rk4")
    n3 = result.y[:, 2]
    assert n3[0] > 2.0
    assert np.min(n3) < -0.5  # a substantial excursion to negative spin


def test_rattleback_energy_decays_overall():
    """Despite the transient internal instability that drives the
    reversal, the model is net dissipative (linear friction `mu` on all
    three components): the quadratic 'energy' diagnostic averaged over the
    first vs. last portion of a long run should show substantial decay."""
    system = Rattleback([0.01, 0.01, 3.0])
    result = system.integrate((0.0, 300.0), dt=5e-3, method="rk4")
    energies = np.array([system.energy(y) for y in result.y])
    n = len(energies)
    early = np.mean(energies[: n // 20])
    late = np.mean(energies[-n // 20 :])
    assert late < 0.1 * early
