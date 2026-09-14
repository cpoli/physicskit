"""Physics-correctness tests for physicskit.semiclassical.systems.scarring."""

from __future__ import annotations

import numpy as np

from physicskit.semiclassical.systems.scarring import bouncing_ball_energies, scar_enhancement


def test_bouncing_ball_energies_match_infinite_well_of_width_2R():
    R = 0.7
    energies = bouncing_ball_energies(R, n_max=4)
    exact = (np.arange(1, 5) * np.pi) ** 2 / (2 * (2 * R) ** 2)
    assert np.allclose(energies, exact)


def test_scar_enhancement_of_uniform_density_is_one():
    x = np.linspace(-2, 2, 200)
    y = np.linspace(-1, 1, 100)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = np.ones_like(X, dtype=bool)
    uniform = np.ones_like(X)
    eta = scar_enhancement(uniform, X, Y, mask, x0=0.3, half_width=0.2)
    assert abs(eta - 1.0) < 1e-10


def test_scar_enhancement_detects_concentrated_ridge():
    x = np.linspace(-2, 2, 400)
    y = np.linspace(-1, 1, 200)
    X, Y = np.meshgrid(x, y, indexing="ij")
    mask = np.ones_like(X, dtype=bool)
    density = np.exp(-((X - 0.3) ** 2) / (2 * 0.05**2))
    eta = scar_enhancement(density, X, Y, mask, x0=0.3, half_width=0.15)
    assert eta > 5.0
