"""Property-based tests (Hypothesis) for invariants that should hold across
the whole input space, not just the handful of fixed examples the other test
files check -- exact physical/mathematical identities that must hold no
matter what parameters or initial conditions are chosen.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from physicskit.chaos.quantum.maps import QuantumBakersMap, QuantumKickedRotor
from physicskit.chaos.systems.billiards import BunimovichStadium, CircleBilliard, SinaiBilliard
from physicskit.chaos.systems.maps import HenonMap
from physicskit.chaos.utils.metrics import numerical_jacobian_map
from physicskit.chaos.utils.timeseries import delay_embed

ALL_BILLIARDS = [
    lambda: CircleBilliard(radius=1.0),
    lambda: SinaiBilliard(cell_size=2.0, scatterer_radius=0.5),
    lambda: BunimovichStadium(radius=1.0, straight_length=2.0),
]

angles = st.floats(min_value=0.0, max_value=2.0 * np.pi, exclude_max=True)


@settings(deadline=None, max_examples=30)
@given(billiard_idx=st.sampled_from(range(len(ALL_BILLIARDS))), angle=angles)
def test_billiard_speed_is_conserved_for_any_launch_angle(billiard_idx, angle):
    """Specular reflection is elastic: |v| = 1 must hold after every bounce,
    for every billiard shape and every launch angle -- not just the sampled
    angles the example-based tests happen to use."""
    billiard = ALL_BILLIARDS[billiard_idx]()
    pos = billiard.sample_interior_point()
    vel = np.array([np.cos(angle), np.sin(angle)])
    result = billiard.simulate(pos, vel, n_bounces=50)
    speeds = np.hypot(result["vx"], result["vy"])
    np.testing.assert_allclose(speeds, 1.0, atol=1e-8)


@settings(deadline=None, max_examples=30)
@given(billiard_idx=st.sampled_from(range(len(ALL_BILLIARDS))), angle=angles)
def test_billiard_boundary_coordinate_s_stays_within_perimeter(billiard_idx, angle):
    """The arclength coordinate s must always lie in [0, perimeter), for any
    launch angle."""
    billiard = ALL_BILLIARDS[billiard_idx]()
    pos = billiard.sample_interior_point()
    vel = np.array([np.cos(angle), np.sin(angle)])
    result = billiard.simulate(pos, vel, n_bounces=50)
    assert np.all(result["s"] >= 0.0)
    assert np.all(result["s"] <= billiard.perimeter() + 1e-9)


@settings(deadline=None, max_examples=50)
@given(
    a=st.floats(min_value=0.5, max_value=2.0),
    b=st.floats(min_value=-0.9, max_value=0.9, allow_nan=False).filter(lambda v: abs(v) > 1e-3),
    x=st.floats(min_value=-1.0, max_value=1.0),
    y=st.floats(min_value=-1.0, max_value=1.0),
)
def test_henon_map_jacobian_matches_analytic_formula_everywhere(a, b, x, y):
    """The Henon map's Jacobian is exactly [[-2*a*x, 1], [b, 0]] at every
    point and every parameter choice, not just the one example the
    fixed-value test checks."""
    henon = HenonMap(a=a, b=b)
    jac = numerical_jacobian_map(henon, np.array([x, y]))
    expected = np.array([[-2.0 * a * x, 1.0], [b, 0.0]])
    np.testing.assert_allclose(jac, expected, atol=1e-4, rtol=1e-4)


@settings(deadline=None, max_examples=50)
@given(
    n=st.integers(min_value=5, max_value=200),
    dim=st.integers(min_value=1, max_value=4),
    tau=st.integers(min_value=1, max_value=5),
)
def test_delay_embed_shape_formula_holds_for_any_valid_dim_and_tau(n, dim, tau):
    """delay_embed's output row count must always be n - (dim - 1) * tau,
    for any series length, dimension, and delay for which that's positive."""
    n_out = n - (dim - 1) * tau
    if n_out <= 0:
        return
    series = np.arange(n, dtype=np.float64)
    embedded = delay_embed(series, dim=dim, tau=tau)
    assert embedded.shape == (n_out, dim)
    # Each row must be an arithmetic progression with common difference tau,
    # since the series itself is just np.arange(n).
    if dim > 1:
        diffs = np.diff(embedded, axis=1)
        np.testing.assert_allclose(diffs, float(tau))


@settings(deadline=None, max_examples=30)
@given(k=st.floats(min_value=0.0, max_value=10.0), dim=st.integers(min_value=2, max_value=60))
def test_quantum_kicked_rotor_floquet_operator_is_unitary_for_any_k_and_dim(k, dim):
    """The Floquet operator's unitarity is an exact mathematical consequence of
    being built from products of unitary (DFT and diagonal phase) matrices --
    it must hold regardless of the kick strength or Hilbert space dimension."""
    u = QuantumKickedRotor(k=k, dim=dim).floquet_operator()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(dim), atol=1e-8)


@settings(deadline=None, max_examples=30)
@given(
    dim=st.integers(min_value=4, max_value=60),
    alpha=st.floats(min_value=0.1, max_value=0.9, allow_nan=False),
)
def test_quantum_bakers_map_floquet_operator_is_unitary_for_any_dim_and_alpha(dim, alpha):
    """Same unitarity guarantee as the kicked rotor, for any (dim, alpha) pair
    that yields a valid two-block split."""
    n1 = round(alpha * dim)
    if n1 < 1 or n1 > dim - 1:
        return
    u = QuantumBakersMap(dim=dim, alpha=alpha).floquet_operator()
    np.testing.assert_allclose(u.conj().T @ u, np.eye(dim), atol=1e-8)
