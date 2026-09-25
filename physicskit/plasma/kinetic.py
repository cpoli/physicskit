"""Electrostatic particle-in-cell (PIC) solution of the 1D1V Vlasov-Poisson system.

Rather than discretizing the distribution function :math:`f(x, v, t)` on a
phase-space grid, a PIC code samples it with a finite set of
computational "super-particles" and lets them stream along exact
single-particle orbits, recovering the self-consistent field by depositing
their charge onto a spatial grid and solving Poisson's equation there each
step. This is Vlasov's collisionless equation
:math:`\\partial_t f + v\\,\\partial_x f - (e/m)E\\,\\partial_v f = 0` solved
by the method of characteristics: each particle *is* one characteristic.
The whole pipeline -- :func:`deposit_number_density`,
:func:`solve_poisson_1d`, :func:`interpolate_field`, :func:`pic_step` --
reproduces collisionless (Landau) damping and the two-stream instability
without ever assuming a collision operator, exactly as the underlying
kinetic theory predicts.

Units throughout are the standard PIC-normalized units: electron charge
:math:`e=1`, mass :math:`m_e=1`, vacuum permittivity
:math:`\\varepsilon_0=1`, and equilibrium density :math:`n_0=1`, so that
the electron plasma frequency :math:`\\omega_{pe}=1` and velocities are in
units of the thermal speed :math:`v_{th}`. Ions are a fixed, uniform,
charge-neutralizing background (infinite mass limit).

The charge-deposit and field-interpolation kernels -- called once per
particle per step -- are compiled with Numba, since they dominate the
cost of every PIC time step.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "maxwellian_velocities",
    "landau_damping_ic",
    "langmuir_wave_ic",
    "two_stream_ic",
    "deposit_number_density",
    "interpolate_field",
    "solve_poisson_1d",
    "pic_step",
    "pic_simulate",
    "landau_damping_rate",
]


def maxwellian_velocities(n_particles: int, v_th: float, seed: int = 0) -> np.ndarray:
    """Sample particle velocities from a Maxwellian (Gaussian) distribution.

    Parameters
    ----------
    n_particles : int
        Number of particles to sample.
    v_th : float
        Thermal speed (standard deviation of the Gaussian), in
        normalized velocity units.
    seed : int, default=0
        Seed for the pseudo-random number generator, for reproducibility.

    Returns
    -------
    ndarray, shape (n_particles,)
        Sampled velocities.

    Examples
    --------
    >>> import numpy as np
    >>> v = maxwellian_velocities(4, v_th=1.0, seed=0)
    >>> bool(np.allclose(v, [0.12573022, -0.13210486, 0.64042265, 0.10490012]))
    True
    """
    rng = np.random.default_rng(seed)
    return rng.normal(0.0, v_th, n_particles)


def landau_damping_ic(n_particles: int, L: float, k_mode: float, alpha: float, v_th: float, seed: int = 0) -> tuple:
    """Quiet-start initial condition seeding a single-mode density perturbation for a Landau damping test.

    Displaces an otherwise uniform particle load by
    :math:`\\delta x = (\\alpha/k)\\sin(kx_0)`, which -- since particle
    number is conserved through the displacement's Jacobian -- produces
    exactly the density perturbation :math:`n(x) \\approx n_0(1 -
    \\alpha\\cos(kx))` used in the classic linear Landau damping test
    problem, without the sampling noise a random density draw would add
    on top of the intended signal.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    L : float
        Domain length (periodic), in normalized length units. Choosing
        :math:`L = 2\\pi/k_{mode}` fits exactly one wavelength.
    k_mode : float
        Wavenumber of the seeded perturbation.
    alpha : float
        Perturbation amplitude (:math:`\\alpha \\ll 1` for the linear regime).
    v_th : float
        Thermal speed of the background Maxwellian.
    seed : int, default=0
        Random seed for the velocity sampling.

    Returns
    -------
    x, v : ndarray, shape (n_particles,)
        Particle positions (in :math:`[0, L)`) and velocities.

    See Also
    --------
    pic_simulate : Evolve this initial condition forward in time.
    landau_damping_rate : The analytic decay rate this test is checked against.

    Examples
    --------
    >>> import numpy as np
    >>> x, v = landau_damping_ic(4, L=4.0, k_mode=2 * 3.141592653589793 / 4.0, alpha=0.1, v_th=1.0, seed=0)
    >>> bool(np.all((x >= 0) & (x < 4.0)))
    True
    >>> x.shape, v.shape
    ((4,), (4,))
    """
    x0 = np.linspace(0.0, L, n_particles, endpoint=False)
    x = np.mod(x0 + (alpha / k_mode) * np.sin(k_mode * x0), L)
    v = maxwellian_velocities(n_particles, v_th, seed)
    return x, v


def langmuir_wave_ic(n_particles: int, L: float, k_mode: float, alpha: float, v_th: float, seed: int = 0) -> tuple:
    """Small-amplitude single-mode initial condition for a Langmuir (electron plasma) wave.

    Uses the same quiet-start displacement as :func:`landau_damping_ic`,
    :math:`\\delta x=(\\alpha/k)\\sin(kx_0)` (density perturbation
    :math:`n_0(1-\\alpha\\cos(kx))`), and additionally gives each particle
    the coherent velocity :math:`\\delta v = (\\alpha\\,\\omega_{pe}/k)\\sin(kx_0)`
    (:math:`\\omega_{pe}=1` in these normalized units). In cold-fluid
    linear theory the displacement then evolves as
    :math:`\\xi(x_0,t) = (\\alpha/k)\\sin(kx_0)\\,[\\cos\\omega_{pe}t + \\sin\\omega_{pe}t]`:
    a standing wave ringing in place at :math:`\\omega_{pe}`, with
    :math:`\\sqrt2` the amplitude (twice the field energy) of the
    displacement-only start and a :math:`\\pi/4` phase shift.

    How long it rings is set by Landau damping, i.e. by :math:`k\\lambda_D`
    (see :func:`landau_damping_rate`), not by the velocity kick: for
    :math:`k_{mode}\\,v_{th} \\ll \\omega_{pe}` this and
    :func:`landau_damping_ic` both oscillate essentially undamped, while
    for :math:`k\\lambda_D \\gtrsim 0.3` both damp.

    Parameters
    ----------
    n_particles : int
        Number of particles.
    L : float
        Domain length (periodic); :math:`L=2\\pi/k_{mode}` fits one wavelength.
    k_mode : float
        Wavenumber of the seeded standing wave.
    alpha : float
        Perturbation amplitude (:math:`\\alpha \\ll 1`).
    v_th : float
        Thermal speed of the background Maxwellian; keep
        :math:`k_{mode}\\,v_{th} \\ll \\omega_{pe}=1` for the wave to be
        only weakly Landau-damped.
    seed : int, default=0
        Random seed for the thermal velocity sampling.

    Returns
    -------
    x, v : ndarray, shape (n_particles,)
        Particle positions and velocities.

    See Also
    --------
    landau_damping_ic : The companion density-only perturbation (no
        coherent velocity kick); it damps or rings for the same
        :math:`k\\lambda_D` as this one.
    pic_simulate : Evolve this initial condition forward in time.

    Examples
    --------
    >>> import numpy as np
    >>> k = 2 * np.pi / 4.0
    >>> x, v = langmuir_wave_ic(4, L=4.0, k_mode=k, alpha=0.05, v_th=0.05, seed=0)
    >>> bool(np.all((x >= 0) & (x < 4.0)))
    True
    >>> x.shape, v.shape
    ((4,), (4,))
    """
    x0 = np.linspace(0.0, L, n_particles, endpoint=False)
    x = np.mod(x0 + (alpha / k_mode) * np.sin(k_mode * x0), L)
    omega_pe = 1.0
    dv = (alpha * omega_pe / k_mode) * np.sin(k_mode * x0)
    v = maxwellian_velocities(n_particles, v_th, seed) + dv
    return x, v


def two_stream_ic(n_particles: int, L: float, v_drift: float, v_th: float, seed: int = 0) -> tuple:
    """Initial condition for the two-stream instability: two counter-streaming Maxwellian beams.

    Splits the particles into two equal populations drifting at
    :math:`\\pm v_{drift}`, each with thermal spread :math:`v_{th}`, and
    seeds the fastest-growing long-wavelength mode with a small
    density ripple. When :math:`v_{drift}` exceeds the thermal spread by
    enough to make the combined velocity distribution doubly-peaked, the
    positive-slope region between the two peaks violates the (kinetic)
    Penrose stability criterion and the ripple grows exponentially,
    eventually rolling the two beams up into a single phase-space vortex.

    Parameters
    ----------
    n_particles : int
        Number of particles (split evenly between the two beams).
    L : float
        Domain length (periodic).
    v_drift : float
        Drift speed of each beam (beams move at :math:`+v_{drift}` and :math:`-v_{drift}`).
    v_th : float
        Thermal spread of each beam.
    seed : int, default=0
        Random seed.

    Returns
    -------
    x, v : ndarray, shape (n_particles,)
        Particle positions and velocities.

    See Also
    --------
    landau_damping_ic : The companion (stable) single-beam initial condition.

    Examples
    --------
    >>> import numpy as np
    >>> x, v = two_stream_ic(1000, L=10.0, v_drift=3.0, v_th=0.5, seed=0)
    >>> x.shape, v.shape
    ((1000,), (1000,))
    >>> bool(np.mean(v) < 0.5)
    True
    """
    rng = np.random.default_rng(seed)
    half = n_particles // 2
    x0 = np.linspace(0.0, L, n_particles, endpoint=False)
    k_seed = 2.0 * np.pi / L
    x = np.mod(x0 + 0.01 * np.sin(k_seed * x0), L)
    v = np.empty(n_particles)
    v[:half] = rng.normal(v_drift, v_th, half)
    v[half:] = rng.normal(-v_drift, v_th, n_particles - half)
    return x, v


@njit(cache=True)
def _deposit_cic_kernel(x, weight, L, ng):
    dx = L / ng
    rho = np.zeros(ng)
    for p in range(x.shape[0]):
        xi = x[p] / dx
        i0 = int(np.floor(xi)) % ng
        frac = xi - np.floor(xi)
        i1 = (i0 + 1) % ng
        rho[i0] += weight * (1.0 - frac) / dx
        rho[i1] += weight * frac / dx
    return rho


@njit(cache=True)
def _interpolate_cic_kernel(x, field, L, ng):
    dx = L / ng
    out = np.empty(x.shape[0])
    for p in range(x.shape[0]):
        xi = x[p] / dx
        i0 = int(np.floor(xi)) % ng
        frac = xi - np.floor(xi)
        i1 = (i0 + 1) % ng
        out[p] = field[i0] * (1.0 - frac) + field[i1] * frac
    return out


@njit(cache=True)
def _push_kernel(x, v, E_at_particles, qm, dt, L):
    n = x.shape[0]
    for p in range(n):
        v[p] += qm * E_at_particles[p] * dt
        x[p] = (x[p] + v[p] * dt) % L
    return x, v


def deposit_number_density(x: np.ndarray, L: float, ng: int, n0: float = 1.0) -> np.ndarray:
    """Deposit particle positions onto a grid as a number density, via cloud-in-cell (CIC) weighting.

    Each particle represents a "cloud" of physical charge spanning one
    grid cell, split linearly between its two nearest grid points -- the
    standard first-order PIC weighting scheme, chosen because it is exact
    for a uniform density and (unlike nearest-grid-point deposit)
    produces a smooth, differentiable force with no self-force
    discontinuities as particles cross cell boundaries.

    Parameters
    ----------
    x : ndarray, shape (n_particles,)
        Particle positions in :math:`[0, L)`.
    L : float
        Domain length (periodic).
    ng : int
        Number of grid points.
    n0 : float, default=1.0
        Equilibrium number density (sets each particle's statistical weight,
        :math:`n_0 L / n_{particles}`).

    Returns
    -------
    ndarray, shape (ng,)
        Number density on the grid.

    See Also
    --------
    interpolate_field : The companion gather operation.

    Examples
    --------
    Total deposited charge exactly equals the physical charge represented,
    regardless of how the particles are distributed:

    >>> import numpy as np
    >>> x = np.array([0.1, 2.4, 4.9, 7.7])
    >>> rho = deposit_number_density(x, L=10.0, ng=20, n0=2.0)
    >>> dx = 10.0 / 20
    >>> round(float(np.sum(rho) * dx), 8)
    20.0
    """
    n_particles = x.shape[0]
    weight = n0 * L / n_particles
    return _deposit_cic_kernel(np.asarray(x, dtype=float), float(weight), float(L), int(ng))


def interpolate_field(x: np.ndarray, field_grid: np.ndarray, L: float) -> np.ndarray:
    """Interpolate a grid-defined field to particle positions, via cloud-in-cell (CIC) weighting.

    The gather step dual to :func:`deposit_number_density`: using the
    same linear weights for both deposit and gather is what makes the
    PIC method momentum-conserving (no self-force on an isolated particle).

    Parameters
    ----------
    x : ndarray, shape (n_particles,)
        Particle positions in :math:`[0, L)`.
    field_grid : ndarray, shape (ng,)
        Field values on the grid, e.g. from :func:`solve_poisson_1d`.
    L : float
        Domain length (periodic).

    Returns
    -------
    ndarray, shape (n_particles,)
        Field value at each particle's position.

    Examples
    --------
    A particle sitting exactly on a grid node picks up that node's value:

    >>> import numpy as np
    >>> ng = 8
    >>> field_grid = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    >>> x = np.array([2 * (10.0 / ng)])
    >>> interpolate_field(x, field_grid, L=10.0)
    array([3.])
    """
    ng = field_grid.shape[0]
    return _interpolate_cic_kernel(np.asarray(x, dtype=float), np.asarray(field_grid, dtype=float), float(L), int(ng))


def solve_poisson_1d(rho: np.ndarray, L: float) -> np.ndarray:
    """Solve the 1D periodic Poisson equation :math:`dE/dx = \\rho` (normalized :math:`\\varepsilon_0=1`) via FFT.

    Parameters
    ----------
    rho : ndarray, shape (ng,)
        Net charge density on the grid (e.g. ion background minus deposited
        electron density). Its mean is discarded, since a uniform charge
        density has no periodic solution and physically should integrate
        to zero net charge in the box.
    L : float
        Domain length (periodic).

    Returns
    -------
    ndarray, shape (ng,)
        Electric field on the grid.

    See Also
    --------
    deposit_number_density : Supplies the density this solves for.
    interpolate_field : Gathers this field back onto the particles.

    Examples
    --------
    >>> import numpy as np
    >>> ng = 64
    >>> L = 2 * np.pi
    >>> x_grid = np.linspace(0, L, ng, endpoint=False)
    >>> rho = np.sin(x_grid)
    >>> E = solve_poisson_1d(rho, L)
    >>> E_exact = -np.cos(x_grid)
    >>> bool(np.max(np.abs(E - E_exact)) < 1e-10)
    True
    """
    ng = rho.shape[0]
    k = 2.0 * np.pi * np.fft.fftfreq(ng, d=L / ng)
    rho_hat = np.fft.fft(rho - np.mean(rho))
    E_hat = np.zeros_like(rho_hat)
    nonzero = k != 0
    E_hat[nonzero] = -1j * rho_hat[nonzero] / k[nonzero]
    return np.real(np.fft.ifft(E_hat))


def pic_step(x: np.ndarray, v: np.ndarray, L: float, ng: int, dt: float, qm: float = -1.0, n0: float = 1.0) -> tuple:
    """Advance the electrostatic PIC system one leapfrog step.

    Deposits the electron density, solves for the self-consistent field
    against a uniform neutralizing ion background, gathers the field back
    onto the particles, and kicks/drifts them -- one full cycle of the
    deposit-solve-gather-push loop at the heart of every PIC code.
    Velocities are staggered a half step behind positions (standard
    leapfrog); see :func:`pic_simulate` for a driver that initializes
    that offset correctly.

    Parameters
    ----------
    x : ndarray, shape (n_particles,)
        Particle positions in :math:`[0, L)`.
    v : ndarray, shape (n_particles,)
        Particle velocities, staggered a half step behind ``x``.
    L : float
        Domain length (periodic).
    ng : int
        Number of grid points.
    dt : float
        Time step.
    qm : float, default=-1.0
        Charge-to-mass ratio in normalized units (``-1.0`` for electrons
        with a fixed, uniform ion background of density ``n0``).
    n0 : float, default=1.0
        Equilibrium number density.

    Returns
    -------
    x_new, v_new : ndarray
        Updated positions and velocities.
    field_energy : float
        :math:`\\int E^2/2\\,dx`, evaluated at the field used for this step's kick.

    See Also
    --------
    pic_simulate : Repeated application of this step with correct leapfrog initialization.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 10, 50, endpoint=False)
    >>> v = np.zeros(50)
    >>> x_new, v_new, fe = pic_step(x, v, L=10.0, ng=32, dt=0.1)
    >>> x_new.shape, v_new.shape
    ((50,), (50,))
    >>> bool(np.isfinite(fe))
    True
    """
    n_e = deposit_number_density(x, L, ng, n0)
    rho = n0 - n_e
    E = solve_poisson_1d(rho, L)
    field_energy = 0.5 * np.sum(E**2) * (L / ng)
    E_at_particles = interpolate_field(x, E, L)
    x_new, v_new = _push_kernel(np.asarray(x, dtype=float).copy(), np.asarray(v, dtype=float).copy(), E_at_particles, float(qm), float(dt), float(L))
    return x_new, v_new, float(field_energy)


def pic_simulate(x0: np.ndarray, v0: np.ndarray, L: float, ng: int, dt: float, steps: int, qm: float = -1.0, n0: float = 1.0) -> dict:
    """Run an electrostatic PIC simulation forward in time, recording the field-energy history.

    Correctly initializes the leapfrog velocity offset (staggering
    ``v0`` back by half a step using the field at ``x0``) before
    repeatedly applying :func:`pic_step`.

    Parameters
    ----------
    x0 : ndarray, shape (n_particles,)
        Initial particle positions, e.g. from :func:`landau_damping_ic` or :func:`two_stream_ic`.
    v0 : ndarray, shape (n_particles,)
        Initial particle velocities.
    L : float
        Domain length (periodic).
    ng : int
        Number of grid points.
    dt : float
        Time step.
    steps : int
        Number of steps to advance.
    qm : float, default=-1.0
        Charge-to-mass ratio in normalized units.
    n0 : float, default=1.0
        Equilibrium number density.

    Returns
    -------
    dict
        ``{"t": ndarray of shape (steps,), "field_energy": ndarray of shape (steps,),
        "x": final positions, "v": final velocities}``.

    See Also
    --------
    pic_step : The single-step update repeated here.
    landau_damping_rate : Analytic decay rate to compare the field-energy history against.

    Examples
    --------
    >>> import numpy as np
    >>> x0, v0 = landau_damping_ic(2000, L=4 * np.pi, k_mode=0.5, alpha=0.01, v_th=1.0, seed=0)
    >>> result = pic_simulate(x0, v0, L=4 * np.pi, ng=32, dt=0.1, steps=20)
    >>> result["field_energy"].shape
    (20,)
    >>> bool(np.all(np.isfinite(result["field_energy"])))
    True
    """
    x = np.asarray(x0, dtype=float).copy()
    v = np.asarray(v0, dtype=float).copy()
    n_e = deposit_number_density(x, L, ng, n0)
    E0 = solve_poisson_1d(n0 - n_e, L)
    E0_at_particles = interpolate_field(x, E0, L)
    v = v - 0.5 * qm * E0_at_particles * dt

    field_energy = np.empty(steps)
    for s in range(steps):
        x, v, fe = pic_step(x, v, L, ng, dt, qm, n0)
        field_energy[s] = fe
    t = np.arange(steps) * dt
    return {"t": t, "field_energy": field_energy, "x": x, "v": v}


def landau_damping_rate(k: float, v_th: float, omega_pe: float = 1.0) -> float:
    """Analytic linear Landau damping rate for a Maxwellian electron plasma.

    The classic weak-damping result (Landau, 1946) for a wave of
    wavenumber :math:`k` on a Maxwellian of thermal speed :math:`v_{th}`,
    with Debye length :math:`\\lambda_D = v_{th}/\\omega_{pe}`:

    .. math::

       \\gamma = -\\omega_{pe}\\sqrt{\\frac{\\pi}{8}}\\,
       \\frac{1}{(k\\lambda_D)^3}\\,
       \\exp\\!\\left(-\\frac{1}{2(k\\lambda_D)^2} - \\frac{3}{2}\\right).

    Electrons resonant with the wave's phase velocity
    (:math:`v = \\omega/k`) surf it, extracting energy from the field on
    net because the Maxwellian has slightly more slower particles being
    accelerated than faster particles being decelerated -- a purely
    collisionless damping mechanism with no dissipation at the particle
    level, reproduced here by :func:`pic_simulate` without any explicit
    damping term in the equations of motion.

    Parameters
    ----------
    k : float
        Wavenumber, in units of inverse Debye length times
        :math:`k\\lambda_D` conventions -- concretely, pass the physical
        wavenumber and set ``v_th``/``omega_pe`` consistently.
    v_th : float
        Electron thermal speed.
    omega_pe : float, default=1.0
        Electron plasma frequency (``1.0`` in the normalized units used
        throughout this module).

    Returns
    -------
    float
        Damping rate :math:`\\gamma` (negative, since the wave decays).
        The formula is only accurate for weak damping,
        :math:`k\\lambda_D \\lesssim 0.5`.

    See Also
    --------
    pic_simulate : Numerically reproduces this decay from first principles.

    Examples
    --------
    The standard textbook benchmark case, :math:`k\\lambda_D = 0.5`:

    >>> round(float(landau_damping_rate(k=0.5, v_th=1.0)), 4)
    -0.1514
    """
    lambda_D = v_th / omega_pe
    klD = k * lambda_D
    return -omega_pe * np.sqrt(np.pi / 8.0) * (1.0 / klD**3) * np.exp(-1.0 / (2.0 * klD**2) - 1.5)
