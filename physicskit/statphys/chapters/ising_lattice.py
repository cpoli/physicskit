"""Lattice spin models: the 2D Ising model, the q-state Potts model, and the XY model.

Three user-facing classes wrap the Numba kernels in
:mod:`physicskit.statphys.core.monte_carlo`:

- :class:`Ising2D` -- the archetypal discrete ferromagnet, with an exactly
  known critical temperature and both Metropolis and Wolff dynamics.
- :class:`PottsModel2D` -- its :math:`q`-state generalization, whose
  transition sharpens from second order (:math:`q \\le 4`) to first order
  (:math:`q > 4`) on the square lattice.
- :class:`XYModel2D` -- continuous planar spins hosting the topological
  Kosterlitz-Thouless transition, driven by vortex-antivortex unbinding
  rather than symmetry breaking.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.monte_carlo import (
    ising_total_energy,
    ising_total_magnetization,
    metropolis_sweep_ising,
    metropolis_sweep_potts,
    metropolis_sweep_xy,
    potts_total_energy,
    seed_numba_random,
    wolff_step_ising,
    xy_plaquette_vorticity,
    xy_total_energy,
)
from physicskit.statphys.utils.thermodynamics import specific_heat, susceptibility

__all__ = ["Ising2D", "PottsModel2D", "XYModel2D"]


class Ising2D:
    """The 2D ferromagnetic Ising model on a periodic :math:`L \\times L` lattice.

    Parameters
    ----------
    L : int, default=32
        Linear lattice size.
    J : float, default=1.0
        Ferromagnetic coupling constant.
    kB : float, default=1.0
        Boltzmann constant, in whatever unit system ``J`` and ``T`` are
        expressed in.
    seed : int, optional
        Seed for reproducible initial spin configurations and Metropolis
        dynamics.

    Attributes
    ----------
    spins : ndarray of shape (L, L)
        Current spin configuration, values in ``{-1, +1}``.

    Examples
    --------
    >>> model = Ising2D(L=16, seed=0)
    >>> model.sweep(beta=1.0, n_sweeps=50)
    >>> -1.0 <= model.magnetization() / model.n_sites <= 1.0
    True
    """

    def __init__(self, L=32, J=1.0, kB=1.0, seed=None):
        self.L = L
        self.J = J
        self.kB = kB
        self.n_sites = L * L
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.spins = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=(L, L))

    @property
    def T_C(self):
        """Onsager's exact critical temperature, :math:`2J / (k_B \\ln(1 + \\sqrt{2})) \\approx 2.269 J / k_B`."""
        return 2.0 * self.J / (self.kB * np.log(1.0 + np.sqrt(2.0)))

    def reset(self, ordered=False):
        """Reinitialize the spin configuration.

        Parameters
        ----------
        ordered : bool, default=False
            If True, start from a fully aligned (all ``+1``) configuration
            (a low-temperature / :math:`T=0` state). Otherwise draw an
            independent random ``\\pm 1`` configuration (a :math:`T=\\infty`
            state).
        """
        if ordered:
            self.spins = np.ones((self.L, self.L), dtype=np.int64)
        else:
            self.spins = self._rng.choice(np.array([-1, 1], dtype=np.int64), size=(self.L, self.L))

    def sweep(self, beta, algorithm="metropolis", n_sweeps=1):
        """Advance the lattice by one or more Monte Carlo sweeps.

        Parameters
        ----------
        beta : float
            Inverse temperature :math:`1/(k_B T)`.
        algorithm : {"metropolis", "wolff"}, default="metropolis"
            Update rule. Metropolis flips single spins; Wolff grows and flips
            whole clusters, which decorrelates much faster near :math:`T_c`.
        n_sweeps : int, default=1
            Number of sweeps (Metropolis) or cluster updates (Wolff) to
            perform.

        Returns
        -------
        int or None
            For ``"wolff"``, the size of the last cluster flipped; ``None``
            for ``"metropolis"``.
        """
        last_cluster = None
        for _ in range(n_sweeps):
            if algorithm == "metropolis":
                metropolis_sweep_ising(self.spins, beta, self.J)
            elif algorithm == "wolff":
                last_cluster = wolff_step_ising(self.spins, beta, self.J)
            else:
                raise ValueError("algorithm must be 'metropolis' or 'wolff'")
        return last_cluster

    def energy(self):
        """Total energy of the current configuration, :math:`-J \\sum_{\\langle i,j\\rangle} s_i s_j`."""
        return ising_total_energy(self.spins, self.J)

    def magnetization(self):
        """Total magnetization :math:`M = \\sum_i s_i` of the current configuration."""
        return ising_total_magnetization(self.spins)

    def run_temperature_sweep(self, temperatures, n_equil=200, n_measure=500, algorithm="metropolis", measure_every=1):
        """Sweep over a range of temperatures, measuring equilibrium thermodynamics at each.

        At every temperature the lattice is first equilibrated for
        ``n_equil`` sweeps, then sampled every ``measure_every`` sweeps for
        ``n_measure`` measurements, from which the specific heat and magnetic
        susceptibility are estimated via their fluctuation-dissipation
        relations. The final configuration at each temperature seeds the next
        (simulated annealing), which shortens equilibration as the sweep
        proceeds.

        Parameters
        ----------
        temperatures : array_like
            Temperatures to sample, typically ordered from high to low to
            exploit annealing.
        n_equil : int, default=200
            Equilibration sweeps discarded before measuring at each
            temperature.
        n_measure : int, default=500
            Number of measurements collected at each temperature.
        algorithm : {"metropolis", "wolff"}, default="metropolis"
            Update rule, see :meth:`sweep`.
        measure_every : int, default=1
            Number of sweeps between successive measurements.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"T"``, ``"E"``, ``"M"``, ``"C_v"``, ``"chi"``, each an
            array aligned with ``temperatures``. ``"E"`` and ``"M"`` are
            per-site averages; ``"C_v"`` and ``"chi"`` are per-site.

        Examples
        --------
        >>> model = Ising2D(L=12, seed=1)
        >>> result = model.run_temperature_sweep(
        ...     [3.5, 2.269, 1.0], n_equil=20, n_measure=20
        ... )
        >>> sorted(result.keys())
        ['C_v', 'E', 'M', 'T', 'chi']
        """
        temperatures = np.asarray(temperatures, dtype=np.float64)
        n_T = len(temperatures)
        E_mean = np.empty(n_T)
        M_mean = np.empty(n_T)
        C_v = np.empty(n_T)
        chi = np.empty(n_T)

        for k, T in enumerate(temperatures):
            beta = 1.0 / (self.kB * T)
            self.sweep(beta, algorithm=algorithm, n_sweeps=n_equil)

            energies = np.empty(n_measure)
            magnetizations = np.empty(n_measure)
            for m in range(n_measure):
                self.sweep(beta, algorithm=algorithm, n_sweeps=measure_every)
                energies[m] = self.energy()
                magnetizations[m] = self.magnetization()

            E_mean[k] = energies.mean() / self.n_sites
            M_mean[k] = np.abs(magnetizations).mean() / self.n_sites
            C_v[k] = specific_heat(energies, T, self.n_sites, kB=self.kB)
            chi[k] = susceptibility(np.abs(magnetizations), T, self.n_sites, kB=self.kB)

        return {"T": temperatures, "E": E_mean, "M": M_mean, "C_v": C_v, "chi": chi}


class PottsModel2D:
    """The 2D ferromagnetic q-state Potts model on a periodic :math:`L \\times L` lattice.

    Each site carries a discrete state :math:`s_i \\in \\{0, \\ldots, q-1\\}`,
    and the Hamiltonian rewards neighboring sites for matching:
    :math:`H = -J \\sum_{\\langle i,j \\rangle} \\delta(s_i, s_j)`. The Ising
    model is the special case :math:`q = 2` (with a rescaled coupling). On the
    square lattice the transition is second order for :math:`q \\le 4` and
    first order for :math:`q > 4`.

    Parameters
    ----------
    L : int, default=32
        Linear lattice size.
    q : int, default=3
        Number of Potts states.
    J : float, default=1.0
        Coupling constant.
    kB : float, default=1.0
        Boltzmann constant.
    seed : int, optional
        Seed for reproducible dynamics.

    Attributes
    ----------
    spins : ndarray of shape (L, L)
        Current state configuration, values in ``{0, ..., q-1}``.
    """

    def __init__(self, L=32, q=3, J=1.0, kB=1.0, seed=None):
        self.L = L
        self.q = q
        self.J = J
        self.kB = kB
        self.n_sites = L * L
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.spins = self._rng.integers(0, q, size=(L, L)).astype(np.int64)

    @property
    def T_C(self):
        """Exact critical temperature of the square-lattice Potts model, :math:`J / (k_B \\ln(1 + \\sqrt{q}))`."""
        return self.J / (self.kB * np.log(1.0 + np.sqrt(self.q)))

    def sweep(self, beta, n_sweeps=1):
        """Advance the lattice by ``n_sweeps`` Metropolis-Hastings sweeps.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_sweeps : int, default=1
            Number of sweeps to perform.
        """
        for _ in range(n_sweeps):
            metropolis_sweep_potts(self.spins, beta, self.J, self.q)

    def energy(self):
        """Total energy :math:`-J \\sum_{\\langle i,j\\rangle} \\delta(s_i, s_j)` of the current configuration."""
        return potts_total_energy(self.spins, self.J)

    def order_parameter(self):
        """Potts order parameter, the rescaled fraction of sites in the majority state.

        .. math::

            m = \\frac{q \\cdot n_{\\max} / N - 1}{q - 1}

        where :math:`n_{\\max}` is the largest population among the ``q``
        states. ``m`` is ``0`` in the fully disordered phase and ``1`` in the
        fully ordered phase.

        Returns
        -------
        float
        """
        counts = np.bincount(self.spins.ravel(), minlength=self.q)
        n_max = counts.max()
        return (self.q * n_max / self.n_sites - 1.0) / (self.q - 1.0)

    def run_temperature_sweep(self, temperatures, n_equil=200, n_measure=500, measure_every=1):
        """Sweep over temperatures, measuring energy, order parameter, and specific heat.

        See :meth:`Ising2D.run_temperature_sweep` for the equilibration and
        measurement protocol; this follows the same pattern.

        Parameters
        ----------
        temperatures : array_like
            Temperatures to sample.
        n_equil : int, default=200
            Equilibration sweeps at each temperature.
        n_measure : int, default=500
            Measurements collected at each temperature.
        measure_every : int, default=1
            Sweeps between successive measurements.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"T"``, ``"E"``, ``"m"``, ``"C_v"``.
        """
        temperatures = np.asarray(temperatures, dtype=np.float64)
        n_T = len(temperatures)
        E_mean = np.empty(n_T)
        m_mean = np.empty(n_T)
        C_v = np.empty(n_T)

        for k, T in enumerate(temperatures):
            beta = 1.0 / (self.kB * T)
            self.sweep(beta, n_sweeps=n_equil)

            energies = np.empty(n_measure)
            orders = np.empty(n_measure)
            for i in range(n_measure):
                self.sweep(beta, n_sweeps=measure_every)
                energies[i] = self.energy()
                orders[i] = self.order_parameter()

            E_mean[k] = energies.mean() / self.n_sites
            m_mean[k] = orders.mean()
            C_v[k] = specific_heat(energies, T, self.n_sites, kB=self.kB)

        return {"T": temperatures, "E": E_mean, "m": m_mean, "C_v": C_v}


class XYModel2D:
    """The 2D classical XY model on a periodic :math:`L \\times L` lattice.

    Each site carries a planar spin angle :math:`\\theta_i \\in [0, 2\\pi)`,
    with Hamiltonian :math:`H = -J \\sum_{\\langle i,j \\rangle}
    \\cos(\\theta_i - \\theta_j)`. The Mermin-Wagner theorem forbids
    conventional long-range order in 2D, but the model still hosts the
    topological Kosterlitz-Thouless (KT) transition at
    :math:`T_{\\text{KT}} \\approx 0.893\\, J / k_B`: bound vortex-antivortex
    pairs at low temperature unbind into a free vortex plasma above
    :math:`T_{\\text{KT}}`.

    Parameters
    ----------
    L : int, default=32
        Linear lattice size.
    J : float, default=1.0
        Coupling constant.
    kB : float, default=1.0
        Boltzmann constant.
    seed : int, optional
        Seed for reproducible dynamics.

    Attributes
    ----------
    theta : ndarray of shape (L, L)
        Current spin angles, in radians.
    """

    #: Numerically established Kosterlitz-Thouless temperature, in units of J/kB.
    _T_KT_REDUCED = 0.8929

    def __init__(self, L=32, J=1.0, kB=1.0, seed=None):
        self.L = L
        self.J = J
        self.kB = kB
        self.n_sites = L * L
        self._rng = np.random.default_rng(seed)
        if seed is not None:
            seed_numba_random(seed)
        self.theta = self._rng.uniform(0.0, 2.0 * np.pi, size=(L, L))

    @property
    def T_KT(self):
        """Approximate (numerically established) Kosterlitz-Thouless transition temperature, :math:`\\approx 0.893\\, J / k_B`."""
        return self._T_KT_REDUCED * self.J / self.kB

    def sweep(self, beta, n_sweeps=1, delta=1.0):
        """Advance the lattice by ``n_sweeps`` Metropolis-Hastings sweeps.

        Parameters
        ----------
        beta : float
            Inverse temperature.
        n_sweeps : int, default=1
            Number of sweeps to perform.
        delta : float, default=1.0
            Maximum trial angular step (radians); tune towards an acceptance
            rate near 50% for efficient sampling.
        """
        for _ in range(n_sweeps):
            metropolis_sweep_xy(self.theta, beta, self.J, delta)

    def energy(self):
        """Total energy :math:`-J \\sum_{\\langle i,j\\rangle} \\cos(\\theta_i - \\theta_j)` of the current configuration."""
        return xy_total_energy(self.theta, self.J)

    def magnetization_vector(self):
        """Net planar magnetization vector :math:`(\\sum_i \\cos\\theta_i, \\sum_i \\sin\\theta_i)`."""
        return np.array([np.cos(self.theta).sum(), np.sin(self.theta).sum()])

    def vorticity(self):
        """Topological charge of every plaquette; see :func:`physicskit.statphys.core.monte_carlo.xy_plaquette_vorticity`.

        Returns
        -------
        ndarray of shape (L, L)
            Plaquette vorticity, close to an integer everywhere and nonzero
            (:math:`\\pm 1`) at vortex/antivortex cores.
        """
        return xy_plaquette_vorticity(self.theta)

    def vortex_count(self, threshold=0.5):
        """Count vortices and antivortices by thresholding the plaquette vorticity.

        Parameters
        ----------
        threshold : float, default=0.5
            Minimum ``|vorticity|`` for a plaquette to be counted as hosting
            a topological charge.

        Returns
        -------
        n_vortices : int
            Number of plaquettes with vorticity near ``+1``.
        n_antivortices : int
            Number of plaquettes with vorticity near ``-1``.
        """
        q = self.vorticity()
        n_vortices = int(np.sum(q > threshold))
        n_antivortices = int(np.sum(q < -threshold))
        return n_vortices, n_antivortices
