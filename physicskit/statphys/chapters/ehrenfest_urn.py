"""The Ehrenfest urn model: the simplest illustration of statistical irreversibility.

Loschmidt and Zermelo objected that Boltzmann's H-theorem could not be
right, since microscopically reversible dynamics cannot produce genuinely
irreversible macroscopic behavior (Loschmidt), and any bounded mechanical
system must eventually return arbitrarily close to its initial state
(Zermelo, via Poincare recurrence). The Ehrenfests' 1907 urn model resolves
both objections in the simplest possible setting: the dynamics is exactly
reversible and does recur exactly, yet for any macroscopic number of balls
the expected recurrence time is astronomically long, while the approach to
equilibrium is essentially instantaneous -- reconciling reversible
microphysics with an observed arrow of time.
"""

from __future__ import annotations

import numpy as np
from scipy.special import gammaln

__all__ = ["EhrenfestUrn"]


class EhrenfestUrn:
    """The Ehrenfest urn model of N labeled balls exchanged between two boxes.

    At each time step, one of the ``n_balls`` balls (chosen uniformly at
    random) is moved from whichever box it currently occupies to the other.

    Parameters
    ----------
    n_balls : int, default=100
        Total number of balls, N.
    n_left_init : int, optional
        Initial number of balls in the left box. Defaults to ``n_balls``
        (all balls start in one box, a maximally non-equilibrium state).
    seed : int, optional
        Seed for the ball-selection sequence.

    Attributes
    ----------
    n_left : int
        Current number of balls in the left box.
    """

    def __init__(self, n_balls=100, n_left_init=None, seed=None):
        self.n_balls = n_balls
        self.n_left = n_balls if n_left_init is None else n_left_init
        self._rng = np.random.default_rng(seed)

    def step(self):
        """Move one uniformly random ball to the other box.

        Returns
        -------
        int
            The updated :attr:`n_left`.
        """
        chosen = self._rng.integers(0, self.n_balls)
        if chosen < self.n_left:
            self.n_left -= 1
        else:
            self.n_left += 1
        return self.n_left

    def entropy(self):
        """Boltzmann entropy :math:`S/k_B = \\ln W` of the current macrostate.

        :math:`W = \\binom{N}{n_{\\text{left}}}` counts the ball-labelings
        consistent with the current left-box occupancy; the binomial
        coefficient is evaluated via the log-gamma function
        (``scipy.special.gammaln``) to stay numerically stable for large
        :math:`N`.

        Returns
        -------
        float
        """
        N, n = self.n_balls, self.n_left
        return float(gammaln(N + 1) - gammaln(n + 1) - gammaln(N - n + 1))

    def run(self, n_steps):
        """Run the urn for ``n_steps`` and record its history.

        Parameters
        ----------
        n_steps : int
            Number of steps to simulate.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"t"``, ``"n_left"``, ``"entropy"``, each of length
            ``n_steps + 1`` (including the initial state).

        Examples
        --------
        >>> urn = EhrenfestUrn(n_balls=50, seed=0)
        >>> history = urn.run(200)
        >>> history["n_left"].shape
        (201,)
        >>> int(history["n_left"][0])
        50
        """
        n_left = np.empty(n_steps + 1, dtype=np.int64)
        entropy = np.empty(n_steps + 1, dtype=np.float64)
        n_left[0] = self.n_left
        entropy[0] = self.entropy()
        for t in range(1, n_steps + 1):
            self.step()
            n_left[t] = self.n_left
            entropy[t] = self.entropy()
        return {"t": np.arange(n_steps + 1), "n_left": n_left, "entropy": entropy}
