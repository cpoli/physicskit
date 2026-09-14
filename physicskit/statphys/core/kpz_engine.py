"""Numba-accelerated Restricted Solid-On-Solid (RSOS) growth dynamics.

The RSOS model is the standard discrete lattice realization of the
Kardar-Parisi-Zhang universality class in 1+1 dimensions: a local growth
rule with no free parameters to tune, whose interface width nonetheless
reproduces the continuum KPZ equation's universal roughness, growth, and
dynamic exponents.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = ["run_rsos"]


@njit(cache=True)
def run_rsos(heights, n_attempts, L):
    """Attempt ``n_attempts`` random single-site RSOS deposition events, in place.

    At each attempt, a uniformly random site ``i`` is chosen and its height
    tentatively raised by one. The move is accepted only if it keeps the
    restricted solid-on-solid constraint :math:`|h_i - h_{i\\pm1}| \\le 1`
    satisfied with both periodic neighbors; otherwise the site is left
    unchanged. This single local constraint -- with no explicit surface
    tension or nonlinear term written anywhere -- is enough to place the
    resulting interface in the KPZ universality class.

    Parameters
    ----------
    heights : ndarray of shape (L,), dtype int64
        Interface height profile. Modified in place.
    n_attempts : int
        Number of random deposition attempts to perform. One Monte Carlo
        sweep conventionally corresponds to ``n_attempts = L``, i.e. one
        attempt per site on average.
    L : int
        Number of sites (periodic boundary conditions).

    Returns
    -------
    ndarray of shape (L,)
        The same array passed in.
    """
    for _ in range(n_attempts):
        i = np.random.randint(0, L)
        left = heights[(i - 1) % L]
        right = heights[(i + 1) % L]
        new_h = heights[i] + 1
        if abs(new_h - left) <= 1 and abs(new_h - right) <= 1:
            heights[i] = new_h
    return heights
