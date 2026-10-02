"""Physics-correctness tests for physicskit.semiclassical.core.bogomolny."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.special import jn_zeros, jnp_zeros

from physicskit.semiclassical.core.bogomolny import bogomolny_quantization_function, bogomolny_transfer_operator


def _circle(N, R=1.0):
    phi = 2 * np.pi * np.arange(N) / N
    pts = R * np.column_stack([np.cos(phi), np.sin(phi)])
    return pts, -pts / R, np.full(N, 2 * np.pi * R / N)


def _minima(ks, f):
    i = np.nonzero((f[1:-1] < f[:-2]) & (f[1:-1] < f[2:]) & (f[1:-1] < 0.3))[0] + 1
    return ks[i]


def test_bogomolny_reproduces_disk_dirichlet_levels():
    pts, nrm, ds = _circle(150)
    ks = np.linspace(2.0, 9.0, 1401)
    found = _minima(ks, bogomolny_quantization_function(ks, pts, nrm, ds))
    exact = np.unique(np.concatenate([jn_zeros(m, 4) for m in range(8)]))
    exact = exact[(exact > 2.0) & (exact < 9.0)]
    assert len(found) == len(exact)
    assert np.max(np.abs(found - exact)) < 0.02


def test_bogomolny_neumann_disk_levels():
    # Neumann levels are zeros of J_m'; for m = 0 those are the zeros of J_1.
    pts, nrm, ds = _circle(150)
    ks = np.linspace(6.0, 10.0, 801)
    found = _minima(ks, bogomolny_quantization_function(ks, pts, nrm, ds, boundary="neumann"))
    exact = np.unique(np.concatenate([jnp_zeros(m, 4) for m in range(1, 12)] + [jn_zeros(1, 4)]))
    exact = exact[(exact > 6.0) & (exact < 10.0)]
    assert len(found) == len(exact)
    assert np.max(np.abs(found - exact)) < 0.08


def test_transfer_operator_is_symmetric_and_scales_with_radius():
    pts, nrm, ds = _circle(80)
    T = bogomolny_transfer_operator(pts, nrm, ds, k=6.0)
    assert np.allclose(T, T.T)
    assert np.all(np.diag(T) == 0)
    # Only the dimensionless k R matters.
    pts2, nrm2, ds2 = _circle(80, R=2.0)
    assert np.allclose(bogomolny_transfer_operator(pts2, nrm2, ds2, k=3.0), T)


def test_transfer_operator_rejects_unknown_boundary():
    pts, nrm, ds = _circle(10)
    with pytest.raises(ValueError):
        bogomolny_transfer_operator(pts, nrm, ds, k=1.0, boundary="robin")
