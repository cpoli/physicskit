import numpy as np
from numba import njit

from physicskit.chaos.core.integrators import leapfrog_integrate, rk4_integrate, yoshida4_integrate


@njit
def _harmonic_force(pos, t, params):
    """Restoring force for a simple harmonic oscillator, for integrator tests."""
    omega = params[0]
    return -(omega**2) * pos


@njit
def _harmonic_rhs(state, t, params):
    """RK4-form right-hand side for the same simple harmonic oscillator."""
    omega = params[0]
    out = np.empty(2)
    out[0] = state[1]
    out[1] = -(omega**2) * state[0]
    return out


def _harmonic_energy(pos, vel, omega):
    return 0.5 * vel**2 + 0.5 * omega**2 * pos**2


def test_yoshida4_conserves_energy_better_than_leapfrog():
    """The 4th-order Yoshida integrator should have much smaller energy drift
    than plain (2nd-order) leapfrog at the same step size, since it is built
    from three leapfrog sub-steps composed to raise the order of accuracy."""
    omega = 2.0
    params = np.array([omega])
    pos0, vel0 = np.array([1.0]), np.array([0.0])
    dt, n_steps = 0.05, 5000

    _, pos_lf, vel_lf = leapfrog_integrate(_harmonic_force, pos0, vel0, 0.0, dt, n_steps, params)
    energy_lf = _harmonic_energy(pos_lf[:, 0], vel_lf[:, 0], omega)
    drift_lf = np.max(np.abs(energy_lf - energy_lf[0]) / energy_lf[0])

    _, pos_y4, vel_y4 = yoshida4_integrate(_harmonic_force, pos0, vel0, 0.0, dt, n_steps, params)
    energy_y4 = _harmonic_energy(pos_y4[:, 0], vel_y4[:, 0], omega)
    drift_y4 = np.max(np.abs(energy_y4 - energy_y4[0]) / energy_y4[0])

    assert drift_y4 < drift_lf


def test_yoshida4_energy_drift_does_not_grow_with_integration_time():
    """Being symplectic, Yoshida4's energy drift should stay bounded as the
    integration horizon grows, unlike RK4's secularly-growing drift."""
    omega = 2.0
    params = np.array([omega])
    pos0, vel0 = np.array([1.0]), np.array([0.0])
    dt = 0.05

    drifts = []
    for n_steps in (2000, 20000, 200000):
        _, pos, vel = yoshida4_integrate(_harmonic_force, pos0, vel0, 0.0, dt, n_steps, params)
        energy = _harmonic_energy(pos[:, 0], vel[:, 0], omega)
        drifts.append(np.max(np.abs(energy - energy[0]) / energy[0]))

    # All three drifts should be the same order of magnitude (bounded), not
    # growing proportionally with the 100x increase in integration horizon.
    assert max(drifts) / min(drifts) < 10.0


def test_rk4_energy_drift_grows_with_integration_time():
    """The non-symplectic RK4 integrator's energy drift grows with
    integration horizon, in contrast to the symplectic integrators -- this is
    exactly the failure mode the symplectic integrators exist to avoid."""
    omega = 2.0
    params = np.array([omega])
    state0 = np.array([1.0, 0.0])
    dt = 0.05

    drifts = []
    for n_steps in (2000, 200000):
        _, states = rk4_integrate(_harmonic_rhs, state0, 0.0, dt, n_steps, params)
        energy = _harmonic_energy(states[:, 0], states[:, 1], omega)
        drifts.append(np.max(np.abs(energy - energy[0]) / energy[0]))

    assert drifts[1] > 5.0 * drifts[0]


def test_yoshida4_matches_leapfrog_for_a_single_leapfrog_substep_limit():
    """Sanity check: with a tiny step size, both integrators should agree
    closely on the trajectory of a simple, smooth system."""
    omega = 1.0
    params = np.array([omega])
    pos0, vel0 = np.array([1.0]), np.array([0.0])
    dt, n_steps = 1e-4, 100

    _, pos_lf, _ = leapfrog_integrate(_harmonic_force, pos0, vel0, 0.0, dt, n_steps, params)
    _, pos_y4, _ = yoshida4_integrate(_harmonic_force, pos0, vel0, 0.0, dt, n_steps, params)

    np.testing.assert_allclose(pos_lf, pos_y4, atol=1e-8)
