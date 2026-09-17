import numpy as np
import pytest

from physicskit.relativity.core.tensors import (
    alcubierre_metric,
    christoffel_symbols,
    einstein_tensor,
    flrw_metric,
    kerr_metric_bl,
    reissner_nordstrom_metric,
    ricci_scalar,
    schwarzschild_metric,
)


def test_schwarzschild_metric_signature_and_symmetry():
    coords = np.array([0.0, 10.0, np.pi / 2.0, 0.3])
    g = schwarzschild_metric(coords, M=1.0)
    assert g.shape == (4, 4)
    assert np.allclose(g, g.T)
    assert g[0, 0] < 0.0  # timelike direction
    assert g[1, 1] > 0.0 and g[2, 2] > 0.0 and g[3, 3] > 0.0


def test_kerr_reduces_to_schwarzschild_at_zero_spin():
    coords = np.array([0.0, 10.0, 1.1, 0.3])
    g_kerr = kerr_metric_bl(coords, M=1.0, a=0.0)
    g_schw = schwarzschild_metric(coords, M=1.0)
    assert np.allclose(g_kerr, g_schw, atol=1e-10)


def test_kerr_metric_has_frame_dragging_cross_term():
    coords = np.array([0.0, 10.0, np.pi / 2.0, 0.0])
    g = kerr_metric_bl(coords, M=1.0, a=0.9)
    assert g[0, 3] != 0.0
    assert g[0, 3] == pytest.approx(g[3, 0])


def test_reissner_nordstrom_reduces_to_schwarzschild_at_zero_charge():
    coords = np.array([0.0, 10.0, np.pi / 2.0, 0.3])
    g_rn = reissner_nordstrom_metric(coords, M=1.0, Q=0.0)
    g_schw = schwarzschild_metric(coords, M=1.0)
    assert np.allclose(g_rn, g_schw)


def test_flrw_metric_flat_matter_dominated():
    a_func = lambda t: (1.0 + t) ** (2.0 / 3.0)
    coords = np.array([0.0, 1.0, np.pi / 2.0, 0.0])
    g = flrw_metric(coords, a_func, k=0.0)
    assert g[0, 0] == pytest.approx(-1.0)
    assert g[1, 1] == pytest.approx(a_func(0.0) ** 2)


def test_flrw_metric_closed_and_open_use_sin_and_sinh_of_chi():
    a_func = lambda t: 1.0
    coords = np.array([0.0, 0.5, np.pi / 2.0, 0.0])
    g_closed = flrw_metric(coords, a_func, k=1.0)
    g_open = flrw_metric(coords, a_func, k=-1.0)
    assert g_closed[2, 2] == pytest.approx(a_func(0.0) ** 2 * np.sin(0.5) ** 2)
    assert g_open[2, 2] == pytest.approx(a_func(0.0) ** 2 * np.sinh(0.5) ** 2)


def test_alcubierre_metric_is_flat_far_from_the_bubble():
    coords = np.array([0.0, 1000.0, 1000.0, 1000.0])
    g = alcubierre_metric(coords, v_s=2.0, sigma=8.0, R=1.0)
    minkowski = np.diag([-1.0, 1.0, 1.0, 1.0])
    assert np.allclose(g, minkowski, atol=1e-6)


def test_christoffel_symbols_are_symmetric_in_lower_indices():
    coords = np.array([0.0, 10.0, np.pi / 2.0, 0.0])
    Gamma = christoffel_symbols(schwarzschild_metric, coords, {"M": 1.0})
    assert Gamma.shape == (4, 4, 4)
    assert np.allclose(Gamma, np.transpose(Gamma, (0, 2, 1)), atol=1e-8)


def test_schwarzschild_vacuum_einstein_tensor_is_zero():
    coords = np.array([0.0, 10.0, np.pi / 2.0, 0.3])
    G = einstein_tensor(schwarzschild_metric, coords, {"M": 1.0})
    assert np.max(np.abs(G)) < 1e-2


def test_schwarzschild_ricci_scalar_is_zero():
    coords = np.array([0.0, 8.0, 1.2, 0.0])
    R = ricci_scalar(schwarzschild_metric, coords, {"M": 1.0})
    assert abs(R) < 1e-2
