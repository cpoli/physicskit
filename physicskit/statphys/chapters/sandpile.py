"""The Bak-Tang-Wiesenfeld sandpile: self-organized criticality.

Ordinary critical phenomena require fine-tuning a parameter (temperature,
occupation probability) to a special value. Bak, Tang, and Wiesenfeld's 1987
sandpile model was the first example of a system that tunes *itself* to a
critical state through its own slow-driving, fast-relaxation dynamics --
"self-organized criticality" (SOC) -- and was proposed as a generic
explanation for the ubiquity of power laws in nature (earthquakes, forest
fires, neural avalanches), without needing anyone to tune anything.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.sandpile_engine import topple_to_stability

__all__ = ["BTWSandpile"]


class BTWSandpile:
    """The Bak-Tang-Wiesenfeld sandpile on an L x L grid with open (absorbing) boundaries.

    Grains are added one at a time to a chosen (or random) site; whenever a
    site's height reaches the toppling threshold, it topples, distributing
    one grain to each orthogonal neighbor (grains that fall off the grid's
    edge are permanently lost). A single added grain can trigger a chain
    reaction of topplings -- an avalanche -- whose size has no
    characteristic scale once the pile has self-organized to its critical
    state.

    Parameters
    ----------
    L : int, default=64
        Linear grid size.
    threshold : int, default=4
        Toppling threshold (4 is the standard value on the square lattice).
    seed : int, optional
        Seed for the random drop-site sequence.

    Attributes
    ----------
    heights : ndarray of shape (L, L)
        Current pile heights; always ``< threshold`` everywhere immediately
        after a call to :meth:`add_grain` or :meth:`run`.
    """

    def __init__(self, L=64, threshold=4, seed=None):
        self.L = L
        self.threshold = threshold
        self._rng = np.random.default_rng(seed)
        self.heights = np.zeros((L, L), dtype=np.int64)

    def add_grain(self, site=None):
        """Add one grain and relax the pile to stability.

        Parameters
        ----------
        site : tuple of int, optional
            ``(i, j)`` grid coordinates to drop the grain at. A uniformly
            random site is chosen if not given.

        Returns
        -------
        int
            Avalanche size: the number of individual topplings triggered by
            this grain (``0`` if it did not trigger any toppling).
        """
        if site is None:
            i = self._rng.integers(0, self.L)
            j = self._rng.integers(0, self.L)
        else:
            i, j = site
        self.heights[i, j] += 1
        return topple_to_stability(self.heights, self.threshold)

    def run(self, n_grains, warmup=0):
        """Drop many grains at random sites, recording each one's avalanche size.

        Parameters
        ----------
        n_grains : int
            Number of grains to add and record after the warmup period.
        warmup : int, default=0
            Grains added first, to let the pile reach its self-organized
            critical state, whose avalanche sizes are discarded.

        Returns
        -------
        ndarray of shape (n_grains,), dtype int64
            Avalanche size triggered by each recorded grain.

        Examples
        --------
        >>> pile = BTWSandpile(L=20, seed=0)
        >>> sizes = pile.run(n_grains=500, warmup=500)
        >>> sizes.shape
        (500,)
        """
        for _ in range(warmup):
            self.add_grain()
        sizes = np.empty(n_grains, dtype=np.int64)
        for k in range(n_grains):
            sizes[k] = self.add_grain()
        return sizes

    def total_grains(self):
        """Total number of grains currently on the pile."""
        return int(self.heights.sum())
