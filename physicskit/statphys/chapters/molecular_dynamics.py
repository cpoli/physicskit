"""N-body Lennard-Jones gas: Velocity Verlet dynamics and Boltzmann's H-theorem.

:class:`LennardJonesGas` simulates a 2D periodic gas of Lennard-Jones
particles and lets it relax toward equilibrium from an arbitrary initial
velocity distribution. Along the way it tracks Boltzmann's H-function, whose
monotonic decrease is the classic microscopic demonstration that
time-reversible molecular dynamics nonetheless produces macroscopic
irreversibility (an arrow of time) as the velocity distribution relaxes to
the Maxwell-Boltzmann form.
"""

from __future__ import annotations

import numpy as np

from physicskit.statphys.core.md_engine import lj_forces, velocity_verlet_step

__all__ = ["LennardJonesGas"]


class LennardJonesGas:
    """A 2D N-particle Lennard-Jones gas in a periodic square box.

    Particles are initialized on a lattice (to avoid the divergent
    Lennard-Jones repulsion of random overlapping placements) and given
    velocities drawn from an arbitrary, generally non-equilibrium initial
    distribution, then integrated with the symplectic Velocity Verlet
    scheme.

    Parameters
    ----------
    n_particles : int, default=100
        Number of particles. Rounded up internally so that
        ``ceil(sqrt(n_particles))**2`` particles fit on a square starting
        lattice; the requested count is used for all physical averages.
    box_size : float, default=20.0
        Side length of the square periodic simulation box.
    temperature_init : float, default=1.0
        Nominal temperature used to set the initial velocity scale (see
        ``initial_velocity_distribution``).
    mass : float, default=1.0
        Common particle mass.
    dt : float, default=0.005
        Integration time step.
    epsilon : float, default=1.0
        Lennard-Jones well depth.
    sigma : float, default=1.0
        Lennard-Jones length scale.
    cutoff : float, default=2.5
        Interaction cutoff radius.
    initial_velocity_distribution : {"maxwell_boltzmann", "uniform", "delta"}, default="uniform"
        Shape of the initial (generally non-equilibrium) velocity
        distribution. ``"uniform"`` and ``"delta"`` are useful starting
        points for watching relaxation toward Maxwell-Boltzmann via the
        H-theorem; ``"maxwell_boltzmann"`` starts already at equilibrium.
    kB : float, default=1.0
        Boltzmann constant.
    seed : int, optional
        Seed for the initial configuration.

    Attributes
    ----------
    positions : ndarray of shape (N, 2)
    velocities : ndarray of shape (N, 2)
    forces : ndarray of shape (N, 2)
        Forces at the current ``positions``, kept up to date across calls to
        :meth:`step` so each step costs one force evaluation.
    time : float
        Elapsed simulation time.
    """

    def __init__(
        self,
        n_particles=100,
        box_size=20.0,
        temperature_init=1.0,
        mass=1.0,
        dt=0.005,
        epsilon=1.0,
        sigma=1.0,
        cutoff=2.5,
        initial_velocity_distribution="uniform",
        kB=1.0,
        seed=None,
    ):
        self.n_particles = n_particles
        self.box_size = box_size
        self.mass = mass
        self.dt = dt
        self.epsilon = epsilon
        self.sigma = sigma
        self.cutoff = cutoff
        self.kB = kB
        self.time = 0.0
        self._rng = np.random.default_rng(seed)

        self.positions = self._lattice_positions(n_particles, box_size, self._rng)
        self.velocities = self._initial_velocities(n_particles, temperature_init, mass, kB, initial_velocity_distribution, self._rng)
        self.forces, self._potential_energy = lj_forces(self.positions, box_size, epsilon=epsilon, sigma=sigma, cutoff=cutoff)

    @staticmethod
    def _lattice_positions(n_particles, box_size, rng):
        n_side = int(np.ceil(np.sqrt(n_particles)))
        spacing = box_size / n_side
        xs, ys = np.meshgrid((np.arange(n_side) + 0.5) * spacing, (np.arange(n_side) + 0.5) * spacing)
        lattice = np.column_stack([xs.ravel(), ys.ravel()])
        jitter = rng.uniform(-0.05 * spacing, 0.05 * spacing, size=lattice.shape)
        return (lattice + jitter)[:n_particles]

    @staticmethod
    def _initial_velocities(n_particles, temperature, mass, kB, distribution, rng):
        std = np.sqrt(kB * temperature / mass)
        if distribution == "maxwell_boltzmann":
            v = rng.normal(0.0, std, size=(n_particles, 2))
        elif distribution == "uniform":
            limit = std * np.sqrt(3.0)
            v = rng.uniform(-limit, limit, size=(n_particles, 2))
        elif distribution == "delta":
            speed = std * np.sqrt(2.0)
            angles = rng.uniform(0.0, 2.0 * np.pi, size=n_particles)
            v = speed * np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            raise ValueError("initial_velocity_distribution must be 'maxwell_boltzmann', 'uniform', or 'delta'")
        v -= v.mean(axis=0, keepdims=True)
        return v

    def step(self, n_steps=1):
        """Advance the simulation by ``n_steps`` Velocity Verlet integration steps.

        Parameters
        ----------
        n_steps : int, default=1
            Number of integration steps to perform.
        """
        for _ in range(n_steps):
            self.positions, self.velocities, self.forces, self._potential_energy = velocity_verlet_step(
                self.positions,
                self.velocities,
                self.forces,
                self.dt,
                self.box_size,
                mass=self.mass,
                epsilon=self.epsilon,
                sigma=self.sigma,
                cutoff=self.cutoff,
            )
            self.time += self.dt

    def kinetic_energy(self):
        """Total kinetic energy :math:`\\sum_i \\frac{1}{2} m v_i^2`."""
        return 0.5 * self.mass * np.sum(self.velocities**2)

    def potential_energy(self):
        """Total Lennard-Jones potential energy at the current configuration."""
        return self._potential_energy

    def total_energy(self):
        """Total (kinetic + potential) energy, conserved up to integration error."""
        return self.kinetic_energy() + self.potential_energy()

    def temperature(self):
        """Instantaneous kinetic temperature from equipartition, :math:`T = \\langle m v^2 \\rangle / (d\\, k_B)`.

        With :math:`d=2` spatial dimensions and 2 degrees of freedom removed
        for the fixed center-of-mass momentum.
        """
        dof = 2 * self.n_particles - 2
        return 2.0 * self.kinetic_energy() / (dof * self.kB)

    def speeds(self):
        """Instantaneous particle speeds :math:`|v_i|`."""
        return np.linalg.norm(self.velocities, axis=1)

    def h_function(self, bins=40, v_max=None):
        """Estimate Boltzmann's H-function from the instantaneous speed histogram.

        .. math::

            H(t) = \\int_0^\\infty f(v, t) \\ln f(v, t) \\, dv

        estimated by binning the current particle speeds into a normalized
        density histogram. :math:`H` decreases monotonically (on average) as
        an arbitrary initial velocity distribution relaxes toward the
        Maxwell-Boltzmann equilibrium, which minimizes :math:`H` at fixed
        energy -- the microscopic root of the second law of thermodynamics.

        Parameters
        ----------
        bins : int, default=40
            Number of histogram bins.
        v_max : float, optional
            Upper edge of the speed histogram. Defaults to
            ``1.5 * speeds().max()``.

        Returns
        -------
        float
            The estimated value of :math:`H(t)`.
        """
        speeds = self.speeds()
        if v_max is None:
            v_max = 1.5 * speeds.max() if speeds.max() > 0 else 1.0
        density, edges = np.histogram(speeds, bins=bins, range=(0.0, v_max), density=True)
        widths = np.diff(edges)
        mask = density > 0
        return float(np.sum(density[mask] * np.log(density[mask]) * widths[mask]))

    def run(self, n_steps, steps_per_record=10, bins=40):
        """Integrate forward while recording a time series of thermodynamic observables.

        Parameters
        ----------
        n_steps : int
            Total number of integration steps to perform.
        steps_per_record : int, default=10
            Number of integration steps between recorded samples.
        bins : int, default=40
            Histogram bins used for each :meth:`h_function` estimate.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"time"``, ``"H"``, ``"temperature"``, ``"kinetic_energy"``,
            ``"potential_energy"``, ``"total_energy"``, each a 1D array
            sampled every ``steps_per_record`` steps (including the initial
            state).

        Examples
        --------
        >>> gas = LennardJonesGas(n_particles=64, box_size=15.0, seed=0)
        >>> history = gas.run(n_steps=200, steps_per_record=50)
        >>> history["H"].shape[0]
        5
        """
        n_records = n_steps // steps_per_record + 1
        history = {
            "time": np.empty(n_records),
            "H": np.empty(n_records),
            "temperature": np.empty(n_records),
            "kinetic_energy": np.empty(n_records),
            "potential_energy": np.empty(n_records),
            "total_energy": np.empty(n_records),
        }

        def _record(idx):
            history["time"][idx] = self.time
            history["H"][idx] = self.h_function(bins=bins)
            history["temperature"][idx] = self.temperature()
            history["kinetic_energy"][idx] = self.kinetic_energy()
            history["potential_energy"][idx] = self.potential_energy()
            history["total_energy"][idx] = self.total_energy()

        _record(0)
        for idx in range(1, n_records):
            self.step(steps_per_record)
            _record(idx)

        return history
