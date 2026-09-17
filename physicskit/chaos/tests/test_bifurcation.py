import numpy as np
import pytest

from physicskit.chaos.systems.continuous import DrivenPendulum, Duffing
from physicskit.chaos.systems.maps import LogisticMap
from physicskit.chaos.visualizers.bifurcation import (
    bifurcation_diagram,
    map_bifurcation_sampler,
    stroboscopic_bifurcation_sampler,
)


def test_bifurcation_diagram_flattens_and_repeats_params_correctly():
    def sample_fn(param):
        return np.array([param, param + 1.0])

    params, samples = bifurcation_diagram([0.0, 10.0], sample_fn)
    np.testing.assert_allclose(params, [0.0, 0.0, 10.0, 10.0])
    np.testing.assert_allclose(samples, [0.0, 1.0, 10.0, 11.0])


def test_bifurcation_diagram_with_multiple_threads_matches_sequential():
    def sample_fn(param):
        return np.array([param, param + 1.0])

    param_values = np.linspace(0.0, 10.0, 15)
    params_seq, samples_seq = bifurcation_diagram(param_values, sample_fn, n_jobs=1)
    params_par, samples_par = bifurcation_diagram(param_values, sample_fn, n_jobs=4, show_progress=True)
    np.testing.assert_allclose(params_par, params_seq)
    np.testing.assert_allclose(samples_par, samples_seq)


def test_map_bifurcation_sampler_shows_fixed_point_below_r3():
    """Below r=3, the logistic map converges to a stable fixed point: every
    surviving sample should be (nearly) identical."""
    sampler = map_bifurcation_sampler(lambda r: LogisticMap(r=r), state0=np.array([0.5]), n_transient=200, n_keep=20)
    samples = sampler(2.5)
    assert np.ptp(samples) < 1e-6


def test_map_bifurcation_sampler_shows_period_two_cycle_above_r3():
    """Just above r=3, the logistic map settles into a period-2 cycle: the
    surviving samples should cluster into exactly two distinct values."""
    sampler = map_bifurcation_sampler(lambda r: LogisticMap(r=r), state0=np.array([0.5]), n_transient=500, n_keep=20)
    samples = sampler(3.2)
    unique_values = np.unique(np.round(samples, decimals=4))
    assert unique_values.size == 2


def test_map_bifurcation_sampler_shows_many_values_when_chaotic():
    """At r=3.9 (chaotic), the surviving samples should spread over many
    distinct values rather than clustering into a small cycle."""
    sampler = map_bifurcation_sampler(lambda r: LogisticMap(r=r), state0=np.array([0.5]), n_transient=500, n_keep=50)
    samples = sampler(3.9)
    unique_values = np.unique(np.round(samples, decimals=3))
    assert unique_values.size > 20


def test_bifurcation_diagram_with_map_sampler_over_a_parameter_sweep():
    sampler = map_bifurcation_sampler(lambda r: LogisticMap(r=r), state0=np.array([0.5]), n_transient=200, n_keep=10)
    r_values = np.linspace(2.4, 4.0, 20)
    params, samples = bifurcation_diagram(r_values, sampler)
    # n_keep=10 plus the trajectory's initial state (row 0) surviving the
    # transient slice, times 20 parameter values.
    assert params.shape == samples.shape == (220,)
    assert np.all(np.isfinite(samples))
    assert np.all((samples >= 0.0) & (samples <= 1.0))


def test_stroboscopic_bifurcation_sampler_on_duffing_is_finite_and_correct_shape():
    omega = 1.2
    period = 2.0 * np.pi / omega
    sampler = stroboscopic_bifurcation_sampler(
        lambda gamma: Duffing(delta=0.3, alpha=-1.0, beta=1.0, gamma=gamma, omega=omega),
        state0=np.array([0.1, 0.0]),
        sample_period=period,
        n_transient_periods=20,
        n_keep_periods=5,
        component=0,
        dt=0.05,
    )
    samples = sampler(0.3)
    assert samples.shape == (5,)
    assert np.all(np.isfinite(samples))


def test_stroboscopic_bifurcation_sampler_shows_fixed_point_for_weak_forcing():
    """With no forcing at all, the (linear, damped) oscillator settles to a
    fixed point: every stroboscopic sample should be (nearly) identical."""
    omega = 1.0
    period = 2.0 * np.pi / omega
    sampler = stroboscopic_bifurcation_sampler(
        lambda gamma: Duffing(delta=0.3, alpha=1.0, beta=0.0, gamma=gamma, omega=omega),
        state0=np.array([1.0, 0.0]),
        sample_period=period,
        n_transient_periods=30,
        n_keep_periods=5,
        component=0,
        dt=0.05,
    )
    samples = sampler(0.0)
    assert np.ptp(samples) < 1e-3


@pytest.mark.slow
def test_driven_pendulum_stroboscopic_sampler_shows_period_doubling_amplitude_dependence():
    """The driven pendulum's forcing-amplitude sweep is the classic
    period-doubling demo: a weak forcing amplitude should settle to a
    single stroboscopic value (period-1 locked to the drive), while a
    strong one (the module's default, deep in the chaotic regime) should
    show many distinct surviving values."""
    omega_d = 2.0 / 3.0
    period = 2.0 * np.pi / omega_d

    def make_system(A):
        return DrivenPendulum(damping=0.5, g_over_l=1.0, A=A, omega_d=omega_d)

    sampler = stroboscopic_bifurcation_sampler(
        make_system,
        state0=np.array([0.2, 0.0]),
        sample_period=period,
        n_transient_periods=50,
        n_keep_periods=15,
        component=0,
        dt=0.02,
    )
    weak_samples = sampler(0.5)
    assert np.ptp(weak_samples) < 1e-2

    strong_samples = sampler(1.5)
    unique_values = np.unique(np.round(strong_samples, decimals=2))
    assert unique_values.size > 5
