import numpy as np
import pytest

from physicskit.particle.electroweak import (
    cp_asymmetry,
    higgs_field_rollover,
    higgs_potential,
    higgs_vev,
    meson_decay_rates_cp_eigenstate,
    qed_dsigma_domega_mumu,
    qed_total_cross_section_mumu,
)
from physicskit.particle.scattering import ALPHA_FS


def test_higgs_potential_symmetric_and_zero_at_origin():
    phi = np.linspace(-2, 2, 21)
    assert higgs_potential(0.0, a=1.0, b=1.0) == pytest.approx(0.0)
    assert higgs_potential(phi, a=1.0, b=1.0) == pytest.approx(higgs_potential(-phi, a=1.0, b=1.0))


def test_higgs_vev_is_the_potential_minimum():
    a, b = 2.0, 0.7
    v = higgs_vev(a, b)
    # V'(v) = -2a v + 4 b v^3 should vanish at the true minimum.
    assert -2 * a * v + 4 * b * v**3 == pytest.approx(0.0, abs=1e-8)
    assert higgs_potential(v, a, b) < higgs_potential(0.0, a, b)


def test_higgs_field_rolls_from_unstable_point_into_a_true_vacuum():
    a, b = 1.0, 1.0
    t = np.linspace(0, 60, 600)
    phi, phidot = higgs_field_rollover(1e-3, 0.0, a, b, t_eval=t, damping=0.08)
    v = higgs_vev(a, b)
    assert abs(phi[0]) < 1e-2
    assert abs(abs(phi[-1]) - v) < 0.05
    assert abs(phidot[-1]) < 0.1  # settled (small residual ringing), not still oscillating at full amplitude


def test_higgs_field_symmetric_perturbation_sign_picks_the_side():
    a, b = 1.0, 1.0
    t = np.linspace(0, 60, 600)
    phi_pos, _ = higgs_field_rollover(1e-3, 0.0, a, b, t_eval=t, damping=0.08)
    phi_neg, _ = higgs_field_rollover(-1e-3, 0.0, a, b, t_eval=t, damping=0.08)
    assert phi_pos[-1] > 0
    assert phi_neg[-1] < 0


def test_qed_dsigma_domega_symmetric_forward_backward():
    vals_fwd = qed_dsigma_domega_mumu(np.array([0.3, 0.9]), sqrt_s=10.0)
    vals_bwd = qed_dsigma_domega_mumu(np.array([-0.3, -0.9]), sqrt_s=10.0)
    assert vals_fwd == pytest.approx(vals_bwd)


def test_qed_dsigma_domega_integrates_to_total_cross_section():
    sqrt_s = 15.0
    cos_theta = np.linspace(-1, 1, 4001)
    dsigma = qed_dsigma_domega_mumu(cos_theta, sqrt_s)
    # integral over solid angle: 2*pi (azimuthal) * integral dcos_theta
    sigma_numeric = 2 * np.pi * np.trapezoid(dsigma, cos_theta)
    sigma_formula = qed_total_cross_section_mumu(sqrt_s)
    assert sigma_numeric == pytest.approx(sigma_formula, rel=1e-4)


def test_qed_total_cross_section_matches_alpha_fs_default():
    sqrt_s = 20.0
    expected = 4 * np.pi * ALPHA_FS**2 / (3 * sqrt_s**2)
    assert qed_total_cross_section_mumu(sqrt_s) == pytest.approx(expected)


def test_cp_asymmetry_vanishes_without_cp_violation():
    t = np.linspace(0, 10, 50)
    A = cp_asymmetry(t, delta_m=0.5, gamma_s=1.0, gamma_l=0.1, epsilon=0.0)
    assert A == pytest.approx(np.zeros_like(t))


def test_meson_decay_rates_are_equal_without_cp_violation():
    t = np.linspace(0, 10, 50)
    g, gbar = meson_decay_rates_cp_eigenstate(t, delta_m=0.5, gamma_s=1.0, gamma_l=0.1, epsilon=0.0)
    assert g == pytest.approx(gbar)


def test_cp_asymmetry_nonzero_and_bounded_with_cp_violation():
    t = np.linspace(0, 10, 200)
    eps = 0.002 * np.exp(1j * 0.7)
    A = cp_asymmetry(t, delta_m=0.5, gamma_s=1.0, gamma_l=0.1, epsilon=eps)
    assert np.any(np.abs(A) > 1e-6)
    assert np.all(np.abs(A) <= 1.0 + 1e-8)


def test_cp_asymmetry_decays_at_late_times_when_gamma_s_dominates():
    t_early = np.array([0.1])
    t_late = np.array([50.0])
    eps = 0.01
    A_early = cp_asymmetry(t_early, delta_m=1.0, gamma_s=5.0, gamma_l=0.05, epsilon=eps)
    A_late = cp_asymmetry(t_late, delta_m=1.0, gamma_s=5.0, gamma_l=0.05, epsilon=eps)
    assert abs(A_late[0]) < abs(A_early[0])
