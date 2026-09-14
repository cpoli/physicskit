"""Jarzynski's equality: exact free energies from irreversible, nonequilibrium work.

The second law only bounds the *average* work done driving a system between
two equilibrium states: :math:`\\langle W \\rangle \\ge \\Delta F`, with
equality only in the reversible, infinitely slow limit. Jarzynski (1997)
found a remarkable equality hiding behind this inequality: averaging not
:math:`W` itself but :math:`e^{-\\beta W}` over an ensemble of repeated
*arbitrarily fast*, arbitrarily irreversible realizations of the same
protocol recovers the equilibrium free energy difference exactly,

.. math::

    e^{-\\beta \\Delta F} = \\left\\langle e^{-\\beta W} \\right\\rangle,

no matter how far from equilibrium any individual trajectory is driven.
Jensen's inequality applied to this identity immediately reproduces the
second law, :math:`\\langle W \\rangle \\ge \\Delta F`, as a corollary
rather than an independent postulate -- nonequilibrium work fluctuations
turn out to carry exact equilibrium information, not just a biased,
dissipation-corrupted estimate of it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["JarzynskiHarmonicTrap"]


class JarzynskiHarmonicTrap:
    """A Brownian particle in a harmonic trap dragged at finite speed, verifying the Jarzynski equality.

    A single overdamped particle sits in the potential :math:`U(x,
    \\lambda) = \\frac{1}{2} k (x - \\lambda)^2`, whose center
    :math:`\\lambda` is the control parameter. :meth:`run_protocol` drags
    :math:`\\lambda` linearly from one value to another over a finite time
    :math:`\\tau`, for many independent realizations starting from thermal
    equilibrium, integrating the overdamped Langevin equation

    .. math::

        \\gamma\\, dx = -k(x - \\lambda)\\, dt + \\sqrt{2\\gamma k_B T}\\, dW_t,

    and accumulating each trajectory's thermodynamic work :math:`W =
    \\int \\frac{\\partial U}{\\partial \\lambda}\\, \\dot\\lambda\\, dt =
    -\\int k(x - \\lambda)\\, d\\lambda`. Because the trap's stiffness never
    changes -- only its center moves -- the true equilibrium free energy is
    identical before and after the protocol, :math:`\\Delta F = 0` exactly,
    by translational invariance. This makes the harmonic trap the cleanest
    possible test case: any protocol speed dissipates heat and drives
    :math:`\\langle W \\rangle` strictly positive (the second law), while
    the exponential average of the *same* work samples should recover
    :math:`\\Delta F = 0` regardless of how fast or slow the drag is.

    Parameters
    ----------
    k : float, default=1.0
        Trap stiffness.
    gamma : float, default=1.0
        Friction (drag) coefficient.
    kB : float, default=1.0
        Boltzmann constant.
    T : float, default=1.0
        Temperature.
    seed : int, optional
        Seed for the initial equilibrium sampling and the thermal noise.
    """

    def __init__(self, k=1.0, gamma=1.0, kB=1.0, T=1.0, seed=None):
        self.k = k
        self.gamma = gamma
        self.kB = kB
        self.T = T
        self._rng = np.random.default_rng(seed)

    def run_protocol(self, lambda_0=0.0, lambda_1=3.0, tau=5.0, n_steps=1000, n_trajectories=5000):
        """Drag the trap center from lambda_0 to lambda_1 over time tau, recording each trajectory's work.

        Every trajectory starts from an independent sample of the
        equilibrium Boltzmann distribution at the initial trap position,
        :math:`x_0 \\sim \\mathcal{N}(\\lambda_0, k_B T / k)`, then is
        integrated forward by Euler-Maruyama while the trap center advances
        linearly in time.

        Parameters
        ----------
        lambda_0, lambda_1 : float, default=0.0, 3.0
            Initial and final trap-center positions.
        tau : float, default=5.0
            Protocol duration. Smaller ``tau`` drags the system further
            from equilibrium and dissipates more heat, but should not
            change the Jarzynski free-energy estimate.
        n_steps : int, default=1000
            Number of Euler-Maruyama integration steps.
        n_trajectories : int, default=5000
            Number of independent realizations. The exponential average
            underlying the Jarzynski estimator is dominated by rare,
            low-work trajectories, so it converges slowly (more
            trajectories are needed for a longer or more violent protocol).

        Returns
        -------
        ndarray of shape (n_trajectories,)
            Accumulated work :math:`W` for each realization.
        """
        dt = tau / n_steps
        D = self.kB * self.T / self.gamma
        noise_scale = np.sqrt(2.0 * D * dt)
        centers = np.linspace(lambda_0, lambda_1, n_steps + 1)

        x = self._rng.normal(loc=lambda_0, scale=np.sqrt(self.kB * self.T / self.k), size=n_trajectories)
        work = np.zeros(n_trajectories)

        for step in range(n_steps):
            lam, lam_next = centers[step], centers[step + 1]
            d_lambda = lam_next - lam
            work += -self.k * (x - lam) * d_lambda
            force = -self.k * (x - lam)
            x = x + (force / self.gamma) * dt + noise_scale * self._rng.standard_normal(n_trajectories)

        return work

    def jarzynski_free_energy_estimate(self, work):
        """Estimate :math:`\\Delta F = -k_B T \\ln \\langle e^{-\\beta W} \\rangle` from work samples.

        Uses the log-sum-exp trick for numerical stability, since
        individual :math:`e^{-\\beta W}` terms can otherwise under- or
        overflow long before the average itself does.

        Parameters
        ----------
        work : array_like
            Work samples, e.g. from :meth:`run_protocol`.

        Returns
        -------
        float
            Estimated free energy difference. For :class:`JarzynskiHarmonicTrap`,
            the true value is exactly 0 regardless of the protocol.
        """
        work = np.asarray(work, dtype=np.float64)
        beta = 1.0 / (self.kB * self.T)
        exponents = -beta * work
        m = np.max(exponents)
        log_mean = m + np.log(np.mean(np.exp(exponents - m)))
        return float(-self.kB * self.T * log_mean)
