"""Tests for physicskit.results: the Result container and its adapters."""

import numpy as np
import pytest

import physicskit.constants as const
from physicskit import results
from physicskit.results import Result
from physicskit.units import SI, geometrized_units, quantum_units, statphys_units

SUN = geometrized_units(mass_kg=const.SOLAR_MASS_KG)


# ---------------------------------------------------------------------------
# Container
# ---------------------------------------------------------------------------


def test_names_and_getitem():
    r = Result([0.0, 1.0], [[1.0, 2.0], [3.0, 4.0]], arrays={"energy": [5.0, 5.0]})
    assert r.names == ["times", "states", "energy"]
    np.testing.assert_array_equal(r["states"], [[1.0, 2.0], [3.0, 4.0]])
    assert r["energy"].dtype == np.float64
    with pytest.raises(KeyError):
        r["missing"]


def test_no_times():
    r = Result(None, np.arange(6).reshape(2, 3))
    assert r.names == ["states"]
    with pytest.raises(KeyError):
        r["times"]


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"times": [0.0, 1.0], "states": [1.0, 2.0, 3.0]}, "first axis"),
        ({"times": [0.0, 1.0], "states": 1.0}, "first axis"),
        ({"times": [[0.0, 1.0]], "states": [1.0]}, "1-D"),
        ({"times": None, "states": [1.0], "arrays": {"states": [1.0]}}, "reserved"),
        ({"times": None, "states": [1.0], "arrays": {"a/b": [1.0]}}, "no '/'"),
        ({"times": None, "states": [1.0], "units": {"energy": "energy"}}, "unknown array"),
        ({"times": None, "states": [1.0], "units": {"states": "furlongs"}}, "unknown dimension"),
    ],
)
def test_validation(kwargs, match):
    with pytest.raises(ValueError, match=match):
        Result(**kwargs)


def test_repr_mentions_shapes_and_system():
    r = Result([0.0, 1.0], np.zeros((2, 3)), unit_system=SUN)
    assert repr(r) == "Result(times=(2,), states=(2, 3), unit_system='relativity (G=c=1)')"


def test_convert_to_si_uses_closed_form_scales():
    # 10 M of coordinate time for a solar-mass black hole is 10 G M / c^3.
    r = Result([0.0, 10.0], [[0.0], [2.0]], units={"times": "time", "states": "length"}, unit_system=SUN, arrays={"flag": [1, 0]})
    si = r.to_si()
    assert si.unit_system == SI
    assert si.times[-1] == pytest.approx(10 * const.G * const.SOLAR_MASS_KG / const.C**3, rel=1e-12)
    assert si.states[-1, 0] == pytest.approx(2 * const.G * const.SOLAR_MASS_KG / const.C**2, rel=1e-12)
    np.testing.assert_array_equal(si["flag"], [1, 0])  # no units entry -> unchanged
    # The original is untouched and the round trip recovers it.
    assert r.times[-1] == 10.0
    back = si.convert(SUN)
    np.testing.assert_allclose(back.times, r.times, rtol=1e-12)
    np.testing.assert_allclose(back.states, r.states, rtol=1e-12)


def test_convert_between_natural_systems():
    # Hartree -> k_B * 1 K units: 1 E_h / k_B = 315775 K (CODATA)
    au = quantum_units(mass_kg=const.ELECTRON_MASS, length_m=5.29177210903e-11)
    r = Result(None, [1.0], units={"states": "energy"}, unit_system=au)
    kelvin = r.convert(statphys_units(temperature_k=1.0))
    assert kelvin.states[0] == pytest.approx(315_775.02, rel=1e-6)


def test_convert_without_unit_system_raises():
    r = Result(None, [1.0], units={"states": "energy"})
    with pytest.raises(ValueError, match="no unit_system"):
        r.to_si()


def test_convert_with_no_units_copies():
    r = Result([0.0], [[1.0]])
    si = r.to_si()
    assert si.unit_system == SI
    si.states[0, 0] = 99.0
    assert r.states[0, 0] == 1.0


# ---------------------------------------------------------------------------
# Adapters, fed with real subpackage output
# ---------------------------------------------------------------------------


def test_from_integrator_rk4_harmonic_oscillator():
    from numba import njit

    from physicskit.integrators import rk4_integrate

    @njit
    def rhs(state, t, params):
        return np.array([state[1], -params[0] * state[0]])

    out = rk4_integrate(rhs, np.array([1.0, 0.0]), 0.0, 0.01, 100, np.array([1.0]))
    r = results.from_integrator(out, method="rk4", metadata={"omega": 1.0})
    assert r.metadata == {"source": "physicskit.integrators", "method": "rk4", "omega": 1.0}
    # x(t) = cos t for the unit harmonic oscillator
    assert r.states[-1, 0] == pytest.approx(np.cos(r.times[-1]), abs=1e-8)


def test_from_integrator_leapfrog_stacks_phase_space():
    from numba import njit

    from physicskit.integrators import leapfrog_integrate

    @njit
    def force(pos, t, params):
        return -pos

    t, pos, vel = leapfrog_integrate(force, np.array([1.0, 0.0]), np.array([0.0, 1.0]), 0.0, 0.01, 50, np.zeros(0))
    r = results.from_integrator((t, pos, vel), method="leapfrog")
    assert r.states.shape == (51, 4)
    np.testing.assert_array_equal(r.states[:, :2], pos)
    np.testing.assert_array_equal(r["velocities"], vel)
    assert r.metadata["state_layout"] == "[positions, velocities]"


def test_from_integrator_accepts_chaos_flow_trajectory():
    from physicskit.chaos.systems.continuous import Lorenz

    r = results.from_integrator(Lorenz().trajectory(n_steps=20))
    assert r.times.shape == (21,) and r.states.shape == (21, 3)


def test_from_integrator_rejects_other_tuples():
    with pytest.raises(ValueError, match="got 1 elements"):
        results.from_integrator((np.zeros(3),))


def test_from_simulation_result_classical():
    from physicskit.classical.systems.hamiltonian import HenonHeilesSystem

    sim = HenonHeilesSystem(q0=[0.1, 0.0], p0=[0.0, 0.2]).integrate((0.0, 1.0), dt=0.01)
    r = results.from_simulation_result(sim, metadata={"note": "test"})
    assert r.names == ["times", "states", "q", "p", "energy"]
    np.testing.assert_array_equal(r.states, sim.y)
    np.testing.assert_array_equal(r["energy"], sim.energy)
    assert r.metadata["method"] == sim.method
    assert r.metadata["note"] == "test"
    # Henon-Heiles H = (px^2+py^2)/2 + (x^2+y^2)/2 + x^2 y - y^3/3 at t=0
    assert r["energy"][0] == pytest.approx(0.5 * 0.2**2 + 0.5 * 0.1**2)


def test_from_simulation_result_splits_extra():
    from physicskit.classical.core.base_system import SimulationResult

    sim = SimulationResult(
        t=np.arange(3.0),
        y=np.zeros((3, 2)),
        method="rk4",
        extra={"lyap": np.ones(3), "n_rejected": np.int64(2), "label": "x", "obj": {1, 2}, "q": np.zeros(3)},
    )
    r = results.from_simulation_result(sim)
    assert set(r.arrays) == {"lyap", "q"}  # "q" is free here: SimulationResult.q is None
    assert r.metadata["extra"] == {"n_rejected": 2, "label": "x", "obj": "{1, 2}"}

    sim.q = np.ones((3, 1))
    r = results.from_simulation_result(sim)
    assert set(r.arrays) == {"q", "lyap", "extra_q"}


def test_from_map_orbit_logistic():
    from physicskit.chaos.systems.maps import LogisticMap

    orbit = LogisticMap(r=3.9).trajectory(n_iter=10)
    r = results.from_map_orbit(orbit)
    np.testing.assert_array_equal(r.times, np.arange(11))
    assert r.metadata["time_kind"] == "iteration"
    # x_{n+1} = r x_n (1 - x_n)
    np.testing.assert_allclose(r.states[1:, 0], 3.9 * r.states[:-1, 0] * (1 - r.states[:-1, 0]), rtol=1e-12)


def test_from_eigen_result_infinite_well():
    from physicskit.quantum.core.eigensolvers import NumerovSolver, infinite_well

    eig = NumerovSolver(np.linspace(0.0, 1.0, 401), infinite_well()).solve(n_states=3)
    au = quantum_units(mass_kg=const.ELECTRON_MASS, length_m=1e-9)
    r = results.from_eigen_result(eig, units={"energies": "energy", "x": "length"}, unit_system=au)
    assert r.times is None and r.states.shape == (3, 401)
    # E_n = n^2 pi^2 / 2 for hbar = m = L = 1
    np.testing.assert_allclose(r["energies"], [np.pi**2 / 2 * n**2 for n in (1, 2, 3)], rtol=1e-6)
    # ... which in SI is n^2 pi^2 hbar^2 / (2 m L^2) with L = 1 nm
    ground_si = np.pi**2 * const.HBAR**2 / (2 * const.ELECTRON_MASS * 1e-9**2)
    assert r.to_si()["energies"][0] == pytest.approx(ground_si, rel=1e-6)


def test_from_spectrum_goe():
    import physicskit.rmt as rmt

    spec = rmt.ensembles.GOE(n=5, seed=3).sample(n_samples=4, return_eigenvectors=True)
    r = results.from_spectrum(spec, metadata={"seed": 3})
    np.testing.assert_array_equal(r.states, spec.eigenvalues)
    np.testing.assert_array_equal(r["eigenvectors"], spec.eigenvectors)
    assert r.metadata == {"source": "physicskit.rmt.Spectrum", "n": 5, "beta": 1.0, "ensemble": "GOE", "scale": spec.scale, "seed": 3}


def test_from_spectrum_without_beta_or_eigenvectors():
    from physicskit.rmt.spectrum import Spectrum

    r = results.from_spectrum(Spectrum(eigenvalues=np.zeros((1, 2)), n=2, beta=None, ensemble="X"))
    assert r.metadata["beta"] is None
    assert r.arrays == {}
