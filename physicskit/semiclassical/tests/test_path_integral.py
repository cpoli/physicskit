"""Physics-correctness tests for physicskit.semiclassical.core.path_integral."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.semiclassical.core.path_integral import (
    build_phasor_diagram,
    discretized_action,
    feynman_phasor_partial_sums,
    free_particle_classical_action,
    free_particle_classical_path,
    harmonic_oscillator_classical_action,
    harmonic_oscillator_classical_path,
    sample_random_paths,
)


def test_discretized_action_matches_free_particle_closed_form_exactly():
    # The free-particle classical path is a straight line, so the midpoint-rule
    # discretization is exact (constant velocity, zero potential) for any n_slices.
    x0, xf, T, m = 0.0, 2.0, 1.0, 1.0
    _, x = free_particle_classical_path(x0, xf, T, m, n_slices=200)
    S_discrete = discretized_action(x, dt=T / 200, m=m, V=lambda x: 0.0 * x)
    S_exact = free_particle_classical_action(x0, xf, T, m)
    assert S_discrete == pytest.approx(S_exact, rel=1e-10)


def test_discretized_action_converges_to_harmonic_oscillator_closed_form():
    # Unlike the free particle, the classical path here is curved, so the midpoint
    # rule has O(dt^2) discretization error. At n_slices=200 the achieved relative
    # error is ~2.5e-4 (checked empirically), well inside the 1% tolerance used here.
    x0, xf, T, m, omega = 0.3, -0.7, 2.3, 1.0, 1.1
    n_slices = 200
    _, x = harmonic_oscillator_classical_path(x0, xf, T, m, omega, n_slices)
    V = lambda x: 0.5 * m * omega**2 * x**2
    S_discrete = discretized_action(x, dt=T / n_slices, m=m, V=V)
    S_exact = harmonic_oscillator_classical_action(x0, xf, T, m, omega)
    assert S_discrete == pytest.approx(S_exact, rel=1e-2)


def test_free_particle_classical_path_minimizes_action_among_random_paths():
    # For a free particle the classical (straight-line) path is a true minimum of
    # the action, not merely a stationary point, so every randomly sampled path
    # must have action >= the classical action (within numerical tolerance).
    x0, xf, T, m, n_slices = 0.0, 1.5, 1.0, 1.0, 40
    S_cl = free_particle_classical_action(x0, xf, T, m)
    paths = sample_random_paths(x0, xf, T, n_slices, n_paths=50, sigma=0.4, seed=1)
    dt = T / n_slices
    actions = np.array([discretized_action(p, dt, m, lambda x: 0.0 * x) for p in paths])
    assert np.all(actions >= S_cl - 1e-8)


def test_harmonic_oscillator_classical_action_closer_than_random_perturbation():
    # Unlike the free particle, the harmonic-oscillator classical path is only a
    # stationary point of the action (not necessarily a minimum), so we only check
    # that it is closer to the exact classical action than a randomly perturbed
    # path -- a weaker but still meaningful sanity check.
    x0, xf, T, m, omega, n_slices = 0.2, 0.8, 1.0, 1.0, 1.3, 60
    S_cl = harmonic_oscillator_classical_action(x0, xf, T, m, omega)
    _, x_cl = harmonic_oscillator_classical_path(x0, xf, T, m, omega, n_slices)
    V = lambda x: 0.5 * m * omega**2 * x**2
    dt = T / n_slices
    S_cl_discrete = discretized_action(x_cl, dt, m, V)

    random_paths = sample_random_paths(x0, xf, T, n_slices, n_paths=1, sigma=0.5, seed=3)
    S_random = discretized_action(random_paths[0], dt, m, V)

    assert abs(S_cl_discrete - S_cl) < abs(S_random - S_cl)


@pytest.mark.parametrize(
    "target,func,kwargs",
    [
        ("path", harmonic_oscillator_classical_path, dict(x0=0.0, xf=1.0, m=1.0, omega=1.0, n_slices=10)),
        ("action", harmonic_oscillator_classical_action, dict(x0=0.0, xf=1.0, m=1.0, omega=1.0)),
    ],
)
def test_harmonic_oscillator_raises_at_conjugate_point(target, func, kwargs):
    # omega*T = pi exactly is a conjugate point: the classical path is non-unique.
    T = np.pi / kwargs["omega"]
    with pytest.raises(ValueError):
        func(T=T, **kwargs)


def test_feynman_phasor_partial_sums_matches_hand_computation():
    actions = np.array([0.3, -1.2, 5.0, 0.0])
    hbar = 0.5
    classical_action = 0.1
    paths = np.zeros((actions.shape[0], 3))  # only shape[0] is used by the function

    partial = feynman_phasor_partial_sums(paths, actions, hbar, classical_action=classical_action)

    # Independent hand computation: sort by |action - classical_action| ascending,
    # then cumulatively sum exp(i*action/hbar) in that order, starting from 0.
    order = np.argsort(np.abs(actions - classical_action))
    expected = np.concatenate([[0.0 + 0.0j], np.cumsum(np.exp(1j * actions[order] / hbar))])

    assert np.allclose(partial, expected)
    assert partial.shape == (actions.shape[0] + 1,)
    assert partial[0] == 0.0


def test_phasor_sum_concentrates_near_classical_path_as_hbar_shrinks():
    # The core physics claim of this module: shrinking hbar makes the propagator's
    # support concentrate ever more tightly around the classical path. Quantify
    # this by comparing, at two very different hbar values, what fraction of the
    # final (all-paths) partial-sum magnitude is already captured by just the
    # closest 10% of paths (sorted by proximity to the classical action).
    def frac_from_closest_10_percent(hbar, seed):
        _, _, _, _, partial = build_phasor_diagram(
            x0=0.0,
            xf=1.0,
            T=1.0,
            m=1.0,
            hbar=hbar,
            potential="free",
            n_slices=40,
            n_paths=400,
            sigma=0.5,
            seed=seed,
        )
        n_total = partial.shape[0] - 1
        idx = max(1, round(0.1 * n_total))
        return abs(partial[idx]) / abs(partial[-1])

    # hbar=20 is much larger than the typical |S - S_cl| spread of this ensemble
    # (tens at most), so phases barely wind and the closest 10% contributes about
    # the "trivial" 10% share. hbar=1.0 is much smaller than that spread, so the
    # closest paths reinforce while distant ones increasingly cancel.
    large_hbar, small_hbar = 20.0, 1.0
    seed = 42

    frac_large = frac_from_closest_10_percent(large_hbar, seed)
    frac_small = frac_from_closest_10_percent(small_hbar, seed)

    assert frac_small > frac_large


def test_free_particle_functions_reject_nonpositive_t_and_bad_n_slices():
    with pytest.raises(ValueError):
        free_particle_classical_path(x0=0.0, xf=1.0, T=0.0, m=1.0, n_slices=4)
    with pytest.raises(ValueError):
        free_particle_classical_path(x0=0.0, xf=1.0, T=1.0, m=1.0, n_slices=0)
    with pytest.raises(ValueError):
        free_particle_classical_action(x0=0.0, xf=1.0, T=0.0, m=1.0)


def test_harmonic_oscillator_functions_reject_nonpositive_t_and_bad_n_slices():
    with pytest.raises(ValueError):
        harmonic_oscillator_classical_path(x0=0.0, xf=1.0, T=0.0, m=1.0, omega=1.0, n_slices=4)
    with pytest.raises(ValueError):
        harmonic_oscillator_classical_path(x0=0.0, xf=1.0, T=1.0, m=1.0, omega=1.0, n_slices=0)
    with pytest.raises(ValueError):
        harmonic_oscillator_classical_action(x0=0.0, xf=1.0, T=0.0, m=1.0, omega=1.0)


def test_sample_random_paths_rejects_nonpositive_t_and_bad_counts():
    with pytest.raises(ValueError):
        sample_random_paths(x0=0.0, xf=1.0, T=0.0, n_slices=4, n_paths=2, sigma=0.1, seed=0)
    with pytest.raises(ValueError):
        sample_random_paths(x0=0.0, xf=1.0, T=1.0, n_slices=0, n_paths=2, sigma=0.1, seed=0)
    with pytest.raises(ValueError):
        sample_random_paths(x0=0.0, xf=1.0, T=1.0, n_slices=4, n_paths=0, sigma=0.1, seed=0)


def test_feynman_phasor_partial_sums_rejects_empty_actions_and_zero_hbar():
    with pytest.raises(ValueError):
        feynman_phasor_partial_sums(paths=np.empty((0, 1)), actions=np.array([]), hbar=0.5)
    with pytest.raises(ValueError):
        feynman_phasor_partial_sums(paths=np.zeros((2, 1)), actions=np.array([0.1, 0.2]), hbar=0.0)


def test_build_phasor_diagram_harmonic_potential_matches_classical_path_and_action():
    omega = 1.5
    paths, actions, x_cl, S_cl, partial = build_phasor_diagram(
        x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.1, potential="harmonic", omega=omega, n_slices=10, n_paths=5, sigma=0.1, seed=0
    )
    expected_path = harmonic_oscillator_classical_path(x0=0.0, xf=1.0, T=1.0, m=1.0, omega=omega, n_slices=10)[1]
    expected_action = harmonic_oscillator_classical_action(x0=0.0, xf=1.0, T=1.0, m=1.0, omega=omega)
    assert np.allclose(x_cl, expected_path)
    assert S_cl == pytest.approx(expected_action)
    assert paths.shape[0] == actions.shape[0] == 6
    assert partial.shape == (7,)


def test_build_phasor_diagram_rejects_nonpositive_t_missing_omega_and_bad_potential():
    with pytest.raises(ValueError):
        build_phasor_diagram(x0=0.0, xf=1.0, T=0.0, m=1.0, hbar=0.1, potential="free", n_slices=10, n_paths=5, sigma=0.1, seed=0)
    with pytest.raises(ValueError):
        build_phasor_diagram(x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.1, potential="harmonic", n_slices=10, n_paths=5, sigma=0.1, seed=0)
    with pytest.raises(ValueError):
        build_phasor_diagram(x0=0.0, xf=1.0, T=1.0, m=1.0, hbar=0.1, potential="quartic", n_slices=10, n_paths=5, sigma=0.1, seed=0)
