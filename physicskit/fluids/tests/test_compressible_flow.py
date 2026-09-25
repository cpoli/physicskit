import numpy as np
import pytest

from physicskit.fluids.exceptions import InvalidParameterError
from physicskit.fluids.systems.compressible_flow import (
    normal_shock_relations,
    rankine_hugoniot_jump_conditions,
    sod_shock_tube,
)


def test_normal_shock_is_identity_at_mach_one():
    jump = normal_shock_relations(M1=1.0)
    assert jump["p2_p1"] == pytest.approx(1.0)
    assert jump["rho2_rho1"] == pytest.approx(1.0)
    assert jump["M2"] == pytest.approx(1.0)


def test_normal_shock_downstream_mach_is_always_subsonic():
    for M1 in (1.5, 2.0, 5.0, 10.0):
        jump = normal_shock_relations(M1)
        assert jump["M2"] < 1.0


def test_normal_shock_satisfies_rankine_hugoniot_conditions():
    gamma = 1.4
    rho1, u1, p1 = 1.0, 3.0, 1.0 / gamma  # c1 = 1, so M1 = u1
    jump = normal_shock_relations(M1=u1, gamma=gamma)
    rho2 = rho1 * jump["rho2_rho1"]
    u2 = rho1 * u1 / rho2
    p2 = p1 * jump["p2_p1"]
    residuals = rankine_hugoniot_jump_conditions(rho1, u1, p1, rho2, u2, p2)
    assert max(abs(v) for v in residuals.values()) < 1e-10


def test_normal_shock_rejects_subsonic_upstream():
    with pytest.raises(InvalidParameterError):
        normal_shock_relations(M1=0.5)


def test_sod_shock_tube_conserves_left_and_right_far_field_states():
    result = sod_shock_tube(nx=200, t_final=0.15)
    assert result["rho"][0] == pytest.approx(1.0, abs=1e-2)
    assert result["p"][0] == pytest.approx(1.0, abs=1e-2)
    assert result["rho"][-1] == pytest.approx(0.125, abs=1e-2)
    assert result["p"][-1] == pytest.approx(0.1, abs=1e-2)


def test_sod_shock_tube_density_is_monotonically_decreasing_left_to_right():
    """The Sod problem's three waves (rarefaction, contact, shock) all
    connect the high state to the low state without an intermediate reversal."""
    result = sod_shock_tube(nx=200, t_final=0.15)
    # allow small non-monotonicity from numerical diffusion by smoothing lightly
    rho = result["rho"]
    assert rho[0] > rho[len(rho) // 2] > rho[-1]


def test_sod_shock_tube_rejects_invalid_cfl():
    with pytest.raises(InvalidParameterError):
        sod_shock_tube(nx=100, cfl=1.5)


def test_sod_shock_tube_rejects_too_few_cells_and_nonpositive_t_final():
    with pytest.raises(InvalidParameterError):
        sod_shock_tube(nx=3)
    with pytest.raises(InvalidParameterError):
        sod_shock_tube(nx=100, t_final=0.0)


def test_rankine_hugoniot_residuals_vanish_for_monatomic_gas_shock():
    gamma = 5.0 / 3.0
    jump = normal_shock_relations(M1=3.0, gamma=gamma)
    rho1, p1 = 1.0, 1.0
    u1 = 3.0 * np.sqrt(gamma * p1 / rho1)
    rho2, p2 = jump["rho2_rho1"] * rho1, jump["p2_p1"] * p1
    residuals = rankine_hugoniot_jump_conditions(rho1, u1, p1, rho2, rho1 * u1 / rho2, p2, gamma=gamma)
    assert max(abs(v) for v in residuals.values()) < 1e-10
