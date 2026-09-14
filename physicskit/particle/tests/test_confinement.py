import numpy as np
import pytest

from physicskit.particle.confinement import string_break_chain, string_tension_energy


def test_string_tension_energy_is_linear_in_r():
    assert string_tension_energy(2.0, kappa=0.9) == pytest.approx(1.8)
    assert string_tension_energy(-3.0, kappa=0.5) == pytest.approx(1.5)


def test_string_break_chain_rejects_superluminal_speed():
    with pytest.raises(ValueError):
        string_break_chain(np.array([0.0, 1.0]), v=1.5, kappa=1.0, m_q=1.0)


def test_string_break_chain_break_times_hit_exact_energy_threshold():
    kappa, m_q, v = 1.0, 1.0, 0.5
    fine_t = np.linspace(0, 20, 5000)
    sim = string_break_chain(fine_t, v=v, kappa=kappa, m_q=m_q, n_breaks=3)
    break_times = sim["break_times"]
    assert len(break_times) == 3

    # evaluate exactly at the analytic break times: total stored energy
    # kappa*r must equal 2*m_q*k for the k-th break (k=1,2,3).
    sim_at_breaks = string_break_chain(break_times, v=v, kappa=kappa, m_q=m_q, n_breaks=3)
    for k in range(1, len(break_times) + 1):
        assert sim_at_breaks["energy_total"][k - 1] == pytest.approx(2.0 * m_q * k, abs=1e-8)


def test_string_break_chain_segment_count_increases_monotonically():
    sim = string_break_chain(np.linspace(0, 20, 500), v=0.5, kappa=1.0, m_q=1.0, n_breaks=3)
    assert np.all(np.diff(sim["n_segments"]) >= 0)
    assert sim["n_segments"][0] == 1
    assert sim["n_segments"][-1] <= 4


def test_string_break_chain_energy_per_segment_never_much_exceeds_threshold():
    sim = string_break_chain(np.linspace(0, 20, 2000), v=0.5, kappa=1.0, m_q=1.0, n_breaks=5)
    # each segment's energy share should stay close to (at most a hair above) 2*m_q
    # right up until the *next* break instantaneously resets the count.
    assert np.all(sim["energy_per_segment"] <= 2.0 * sim["m_q"] + 1e-6)
