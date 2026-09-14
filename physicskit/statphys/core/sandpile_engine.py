"""Numba-accelerated toppling dynamics for the Bak-Tang-Wiesenfeld sandpile."""

from __future__ import annotations

from numba import njit

__all__ = ["topple_to_stability"]


@njit(cache=True)
def topple_to_stability(heights, threshold=4):
    """Relax a sandpile height grid to stability by repeated toppling, in place.

    Open boundary conditions: grains toppled off the edge are lost from the
    system. A site with height >= ``threshold`` topples, losing
    ``threshold`` grains and giving one grain to each of its (up to four)
    orthogonal neighbors.

    Parameters
    ----------
    heights : ndarray of shape (L, L), dtype int64
        Grid of pile heights. Modified in place.
    threshold : int, default=4
        Critical height at which a site topples (4 is the standard BTW
        value on the square lattice, one grain per neighbor direction).

    Returns
    -------
    int
        Total number of individual topplings that occurred during this
        relaxation (the avalanche size).
    """
    L0, L1 = heights.shape
    n_topplings = 0
    unstable = True
    while unstable:
        unstable = False
        for i in range(L0):
            for j in range(L1):
                if heights[i, j] >= threshold:
                    unstable = True
                    heights[i, j] -= threshold
                    n_topplings += 1
                    if i > 0:
                        heights[i - 1, j] += 1
                    if i < L0 - 1:
                        heights[i + 1, j] += 1
                    if j > 0:
                        heights[i, j - 1] += 1
                    if j < L1 - 1:
                        heights[i, j + 1] += 1
    return n_topplings
