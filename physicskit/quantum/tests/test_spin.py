"""Tests for physicskit.quantum.chapters.spin: RabiProblem's Omega_R=0
edge case (no drive, no detuning), not exercised by test_animations.py.
"""

from __future__ import annotations

import numpy as np

from physicskit.quantum.chapters.spin import RabiProblem
from physicskit.quantum.core.operators import identity2


def test_rabi_propagator_is_identity_when_undriven_and_on_resonance():
    rabi = RabiProblem(omega0=1.0, omega_d=1.0, Omega=0.0)
    assert rabi.generalized_rabi_frequency == 0.0
    np.testing.assert_allclose(rabi.propagator(t=2.5), identity2)


def test_rabi_excited_state_population_stays_zero_when_undriven():
    rabi = RabiProblem(omega0=1.0, omega_d=1.0, Omega=0.0)
    t = np.linspace(0, 10, 20)
    np.testing.assert_allclose(rabi.excited_state_population(t), np.zeros_like(t))
