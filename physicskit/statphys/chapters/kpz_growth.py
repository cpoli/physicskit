"""The Kardar-Parisi-Zhang equation: universal roughening of growing interfaces.

Kardar, Parisi, and Zhang's 1986 stochastic partial differential equation,

.. math::

    \\frac{\\partial h}{\\partial t} = \\nu \\nabla^2 h
    + \\frac{\\lambda}{2}(\\nabla h)^2 + \\eta(x, t),

describes a growing interface driven by uncorrelated noise :math:`\\eta`,
smoothed by surface tension :math:`\\nu`, and locally accelerated by the
nonlinear slope term :math:`\\lambda(\\nabla h)^2` -- growth normal to the
surface, not straight up. That nonlinear term is what separates the KPZ
universality class from the simpler Edwards-Wilkinson equation
(:math:`\\lambda = 0`): it couples length scales together and produces a
new, nontrivial set of scaling exponents shared by an enormous range of
physical growth processes, from bacterial colonies to burning paper fronts.

Rather than discretize the continuum SPDE directly (whose nonlinear term is
notoriously delicate to finite-difference consistently), this module
implements the Restricted Solid-On-Solid (RSOS) model, an exact discrete
growth automaton with no adjustable surface-tension or coupling parameters
at all, long established (Kim & Kosterlitz, 1989) to lie in the KPZ
universality class.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.kpz_engine import run_rsos
from physicskit.statphys.core.monte_carlo import seed_numba_random

__all__ = ["KPZInterface"]


class KPZInterface:
    """A 1D Restricted Solid-On-Solid growing interface on a periodic lattice of L sites.

    Parameters
    ----------
    L : int, default=256
        Number of sites (periodic boundary conditions).
    seed : int, optional
        Seed for the random deposition-site sequence.

    Attributes
    ----------
    heights : ndarray of shape (L,)
        Current interface height profile, initialized flat at zero.
    time : float
        Elapsed time, in units of Monte Carlo sweeps (one sweep = L
        attempted deposition events).
    """

    def __init__(self, L=256, seed=None):
        self.L = L
        self.heights = np.zeros(L, dtype=np.int64)
        self.time = 0.0
        if seed is not None:
            seed_numba_random(seed)

    def grow(self, n_sweeps=1):
        """Advance the interface by ``n_sweeps`` Monte Carlo sweeps of RSOS deposition.

        Parameters
        ----------
        n_sweeps : float, default=1
            Number of sweeps (``L`` deposition attempts each) to perform.
        """
        n_attempts = int(round(n_sweeps * self.L))
        run_rsos(self.heights, n_attempts, self.L)
        self.time += n_sweeps

    def width(self):
        """Interface width :math:`w = \\sqrt{\\langle (h - \\bar h)^2 \\rangle}`, the key KPZ observable.

        Returns
        -------
        float
        """
        h = self.heights.astype(np.float64)
        return float(np.sqrt(np.mean((h - h.mean()) ** 2)))

    def mean_height(self):
        """Mean interface height :math:`\\bar h`, which grows linearly in time on average."""
        return float(self.heights.mean())

    def run_growth_curve(self, t_max, n_points=40, log_spaced=True):
        """Grow the interface while recording the width at a sequence of checkpoint times.

        The width is expected to grow as :math:`w(t) \\sim t^{\\beta}` with
        :math:`\\beta = 1/3` at early times, crossing over to a
        length-limited saturation :math:`w_{\\text{sat}} \\sim L^{\\alpha}`
        with :math:`\\alpha = 1/2` once :math:`t` exceeds the correlation
        time :math:`\\sim L^{z}` (:math:`z = 3/2`).

        Parameters
        ----------
        t_max : float
            Final time (in sweeps) to grow to.
        n_points : int, default=40
            Number of checkpoints to record.
        log_spaced : bool, default=True
            Use logarithmically spaced checkpoints (recommended for
            resolving the early-time power-law growth regime) rather than
            linearly spaced ones.

        Returns
        -------
        times : ndarray of shape (n_points,)
        widths : ndarray of shape (n_points,)
        """
        if log_spaced:
            checkpoints = np.unique(np.round(np.logspace(0, np.log10(t_max), n_points)).astype(np.int64))
        else:
            checkpoints = np.unique(np.round(np.linspace(1, t_max, n_points)).astype(np.int64))

        times = np.empty(len(checkpoints))
        widths = np.empty(len(checkpoints))
        t_done = 0
        for k, t in enumerate(checkpoints):
            dt = int(t) - t_done
            if dt > 0:
                self.grow(n_sweeps=dt)
                t_done = int(t)
            times[k] = self.time
            widths[k] = self.width()
        return times, widths
