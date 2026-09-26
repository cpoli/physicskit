"""Shared fixtures for the physicskit benchmarks."""

import numpy as np
import pytest
from numba import njit


@njit(cache=True)
def kepler_accel(pos, t, params):
    """Acceleration of a test body around a point mass ``params[0]`` (G = 1)."""
    r = np.sqrt(pos[0] ** 2 + pos[1] ** 2)
    return -params[0] * pos / r**3


@njit(cache=True)
def kepler_rhs(state, t, params):
    """First-order Kepler right-hand side for ``state = (x, y, vx, vy)``."""
    out = np.empty(4)
    out[:2] = state[2:]
    out[2:] = kepler_accel(state[:2], t, params)
    return out


@pytest.fixture
def kepler():
    """Circular-orbit initial conditions (r = 1, v = 1, GM = 1, period 2 pi)."""
    return {
        "pos0": np.array([1.0, 0.0]),
        "vel0": np.array([0.0, 1.0]),
        "params": np.array([1.0]),
        "accel": kepler_accel,
        "rhs": kepler_rhs,
    }
