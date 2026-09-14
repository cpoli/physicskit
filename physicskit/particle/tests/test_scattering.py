import numpy as np
import pytest

from physicskit.particle.kinematics import FourVector
from physicskit.particle.scattering import (
    ALPHA_FS,
    impact_parameter,
    mandelstam_s,
    mandelstam_t,
    mandelstam_u,
    rutherford_dsigma_domega,
)


def test_mandelstam_sum_rule_for_a_2to2_process():
    # elastic head-on-ish scattering: 1+2 -> 3+4, all equal mass m,
    # constructed so that energy-momentum is manifestly conserved.
    m = 0.5
    E = 2.0
    p = np.sqrt(E**2 - m**2)
    p1 = FourVector(E, 0.0, 0.0, p)
    p2 = FourVector(E, 0.0, 0.0, -p)
    # outgoing particles rotated by some angle theta in the CM frame,
    # same energy/momentum magnitude by symmetry (equal masses, elastic).
    theta = 0.7
    p3 = FourVector(E, p * np.sin(theta), 0.0, p * np.cos(theta))
    p4 = FourVector(E, -p * np.sin(theta), 0.0, -p * np.cos(theta))

    s = mandelstam_s(p1, p2)
    t = mandelstam_t(p1, p3)
    u = mandelstam_u(p1, p4)
    mass_sum = p1.mass**2 + p2.mass**2 + p3.mass**2 + p4.mass**2
    assert s + t + u == pytest.approx(mass_sum, abs=1e-8)


def test_mandelstam_s_of_symmetric_pair():
    p1 = FourVector(1.0, 0.0, 0.0, 0.6)
    p2 = FourVector(1.0, 0.0, 0.0, -0.6)
    assert mandelstam_s(p1, p2) == pytest.approx(4.0)


def test_rutherford_functional_form_sin4_law():
    thetas = np.linspace(0.2, 2.5, 8)
    vals = rutherford_dsigma_domega(thetas, Z1=2, Z2=79, E_kin=5.0)
    normalized = vals * np.sin(thetas / 2.0) ** 4
    assert normalized == pytest.approx(normalized[0])


def test_rutherford_diverges_at_zero_angle():
    val = rutherford_dsigma_domega(0.0, Z1=1, Z2=1, E_kin=1.0)
    assert np.isinf(val)


def test_impact_parameter_limits():
    b_head_on = impact_parameter(np.pi, Z1=1, Z2=79, E_kin=5.0)
    assert b_head_on == pytest.approx(0.0, abs=1e-10)

    b_small_angle = impact_parameter(1e-3, Z1=1, Z2=79, E_kin=5.0)
    b_larger_angle = impact_parameter(1e-2, Z1=1, Z2=79, E_kin=5.0)
    assert b_small_angle > b_larger_angle > 0


def test_alpha_fs_value():
    assert ALPHA_FS == pytest.approx(1 / 137.035999084)
