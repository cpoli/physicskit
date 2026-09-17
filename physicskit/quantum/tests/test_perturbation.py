"""Tests for physicskit.quantum.chapters.perturbation: Zeeman and linear
Stark shifts (closed-form formulas), and the Floquet-driven infinite box.
"""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.quantum._compat import trapz
from physicskit.quantum.chapters.perturbation import (
    FloquetDrivenBox,
    linear_stark_shift,
    stark_n2_quartet,
    zeeman_splitting,
    zeeman_spectrum,
)
from physicskit.quantum.core.solvers import SplitOperatorSolver1D


# --- Zeeman -------------------------------------------------------------


def test_zeeman_splitting_matches_formula():
    # Delta E = mu_B * B * (m_l + g_s * m_s)
    assert zeeman_splitting(m_l=1, m_s=0.5, B=2.0, mu_B=1.0, g_s=2.0) == pytest.approx(4.0)
    assert zeeman_splitting(m_l=0, m_s=0.0, B=5.0) == pytest.approx(0.0)


def test_zeeman_splitting_zero_field_has_no_shift():
    assert zeeman_splitting(m_l=3, m_s=-0.5, B=0.0) == pytest.approx(0.0)


def test_zeeman_spectrum_has_2_times_2l_plus_1_sorted_sublevels():
    l = 1
    shifts = zeeman_spectrum(l, B=1.0, mu_B=1.0, g_s=2.0)
    assert shifts.shape == (2 * (2 * l + 1),)
    np.testing.assert_allclose(shifts, np.sort(shifts))
    np.testing.assert_allclose(shifts, [-2.0, -1.0, 0.0, 0.0, 1.0, 2.0])


# --- Linear Stark effect --------------------------------------------------


def test_linear_stark_shift_matches_formula():
    # valid: n1+n2+|m|+1 = n
    shift = linear_stark_shift(n=2, n1=1, n2=0, m=0, F=1.0, a0=1.0)
    assert shift == pytest.approx(1.5 * 2 * (1 - 0) * 1.0 * 1.0)


def test_linear_stark_shift_rejects_invalid_parabolic_numbers():
    with pytest.raises(ValueError):
        linear_stark_shift(n=2, n1=0, n2=0, m=0, F=1.0)


def test_stark_n2_quartet_reproduces_classic_minus3_0_0_plus3_pattern():
    levels = stark_n2_quartet(F=1.0, a0=1.0)
    shifts = [lv.shift for lv in levels]
    assert shifts == sorted(shifts)
    np.testing.assert_allclose(shifts, [-3.0, 0.0, 0.0, 3.0])


def test_stark_n2_quartet_scales_linearly_with_field():
    levels_1 = stark_n2_quartet(F=1.0)
    levels_2 = stark_n2_quartet(F=2.0)
    np.testing.assert_allclose([lv.shift for lv in levels_2], [2 * lv.shift for lv in levels_1])


# --- Floquet-driven box ---------------------------------------------------


@pytest.fixture
def small_box():
    return FloquetDrivenBox(L=1.0, V0=5.0, omega=20.0, n_grid=256)


def test_box_eigenstate_is_normalized_and_vanishes_outside_well(small_box):
    psi = small_box.box_eigenstate(1)
    assert trapz(psi**2, small_box.x) == pytest.approx(1.0, abs=1e-3)
    outside = (small_box.x < 0) | (small_box.x > small_box.L)
    np.testing.assert_allclose(psi[outside], 0.0)


def test_box_energy_matches_infinite_well_formula(small_box):
    for n in (1, 2, 3):
        expected = (n**2 * np.pi**2 * small_box.hbar**2) / (2 * small_box.m * small_box.L**2)
        assert small_box.box_energy(n) == pytest.approx(expected)


def test_make_solver_returns_configured_split_operator_solver(small_box):
    solver = small_box.make_solver(dt=1e-4)
    assert isinstance(solver, SplitOperatorSolver1D)
    np.testing.assert_allclose(solver.x, small_box.x)


def test_transition_probability_starts_near_delta_function(small_box):
    times, probs = small_box.transition_probability(n_initial=1, n_final=1, t_max=1e-3, dt=1e-4)
    assert times.shape == probs.shape
    assert probs[0] == pytest.approx(1.0, abs=1e-2)
    assert np.all(probs >= -1e-9) and np.all(probs <= 1.0 + 1e-9)


def test_transition_probability_off_diagonal_starts_near_zero(small_box):
    _, probs = small_box.transition_probability(n_initial=1, n_final=2, t_max=1e-3, dt=1e-4)
    assert probs[0] == pytest.approx(0.0, abs=1e-2)


def test_floquet_quasienergies_are_real_and_sorted(small_box):
    quasi_energies = small_box.floquet_quasienergies(n_levels=2, dt=1e-3)
    assert quasi_energies.shape == (2,)
    assert np.all(np.isreal(quasi_energies))
    np.testing.assert_allclose(quasi_energies, np.sort(quasi_energies))
