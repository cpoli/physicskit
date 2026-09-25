"""Thermodynamic observables derived from Monte Carlo and molecular dynamics samples.

Every function here takes arrays of samples collected during a simulation
(energies, magnetizations, velocities, spin configurations, ...) and returns a
scalar or array observable, following the standard fluctuation-dissipation
relations of the canonical ensemble.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bec_condensate_fraction",
    "binder_cumulant",
    "bose_einstein_occupation",
    "fermi_dirac_occupation",
    "maxwell_boltzmann_component_pdf",
    "maxwell_boltzmann_speed_pdf",
    "specific_heat",
    "spin_correlation_function",
    "susceptibility",
]


def specific_heat(energies, temperature, n_sites, kB=1.0):
    """Specific heat per site from the energy fluctuations of a canonical sample.

    .. math::

        C_v = \\frac{\\langle E^2 \\rangle - \\langle E \\rangle^2}{k_B T^2 N}

    Parameters
    ----------
    energies : array_like
        Sampled total energies at fixed temperature.
    temperature : float
        Sampling temperature.
    n_sites : int
        Number of lattice sites (or particles) ``N``, used to report an
        intensive, per-site quantity.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    float
        Specific heat per site.

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> E = rng.normal(loc=-100.0, scale=5.0, size=10000)
    >>> bool(specific_heat(E, temperature=2.0, n_sites=64) > 0)
    True
    """
    energies = np.asarray(energies, dtype=np.float64)
    var_E = energies.var()
    return var_E / (kB * temperature**2 * n_sites)


def susceptibility(magnetizations, temperature, n_sites, kB=1.0):
    """Magnetic susceptibility per site from magnetization fluctuations.

    .. math::

        \\chi = \\frac{\\langle M^2 \\rangle - \\langle M \\rangle^2}{k_B T N}

    Parameters
    ----------
    magnetizations : array_like
        Sampled total magnetizations at fixed temperature.
    temperature : float
        Sampling temperature.
    n_sites : int
        Number of lattice sites ``N``.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    float
        Susceptibility per site.
    """
    magnetizations = np.asarray(magnetizations, dtype=np.float64)
    var_M = magnetizations.var()
    return var_M / (kB * temperature * n_sites)


def binder_cumulant(magnetizations):
    """Fourth-order Binder cumulant of a magnetization sample.

    .. math::

        U_4 = 1 - \\frac{\\langle M^4 \\rangle}{3 \\langle M^2 \\rangle^2}

    ``U_4`` is a standard finite-size-scaling diagnostic: it is
    size-independent exactly at the critical temperature, which makes
    crossing plots for several lattice sizes a robust way to locate
    :math:`T_c`.

    Parameters
    ----------
    magnetizations : array_like
        Sampled total (or per-site) magnetizations.

    Returns
    -------
    float
        The Binder cumulant: at most :math:`2/3` (reached by a two-peak
        ordered distribution) and ``0`` for a Gaussian (disordered)
        distribution; heavier-than-Gaussian tails, e.g. near a first-order
        transition, make it negative.
    """
    m = np.asarray(magnetizations, dtype=np.float64)
    m2 = np.mean(m**2)
    m4 = np.mean(m**4)
    return 1.0 - m4 / (3.0 * m2**2)


def spin_correlation_function(spins, max_r=None):
    """Radially averaged equal-time spin-spin correlation function :math:`G(r)`.

    Computed efficiently via the Wiener-Khinchin theorem: the autocorrelation
    of a periodic field equals the inverse FFT of its power spectrum.

    .. math::

        G(r) = \\langle s_0 s_r \\rangle - \\langle s \\rangle^2

    averaged over all site pairs at (periodic, Euclidean-rounded) separation
    ``r``.

    Parameters
    ----------
    spins : ndarray of shape (L, L)
        Spin (or angle-cosine) configuration on a periodic square lattice.
    max_r : int, optional
        Largest separation to report. Defaults to ``L // 2``.

    Returns
    -------
    r : ndarray of shape (max_r,)
        Integer separations ``0, 1, ..., max_r - 1``.
    G : ndarray of shape (max_r,)
        Correlation function averaged over all pairs at each rounded
        separation.
    """
    spins = np.asarray(spins, dtype=np.float64)
    L = spins.shape[0]
    if max_r is None:
        max_r = L // 2

    mean_s = spins.mean()
    fft = np.fft.fft2(spins)
    autocorr = np.fft.ifft2(fft * np.conj(fft)).real / (L * L)

    ii, jj = np.meshgrid(np.arange(L), np.arange(L), indexing="ij")
    di = np.minimum(ii, L - ii)
    dj = np.minimum(jj, L - jj)
    dist = np.sqrt(di**2 + dj**2)
    r_bin = np.round(dist).astype(int)

    G = np.zeros(max_r, dtype=np.float64)
    for r in range(max_r):
        mask = r_bin == r
        if np.any(mask):
            G[r] = autocorr[mask].mean() - mean_s**2
    return np.arange(max_r), G


def maxwell_boltzmann_speed_pdf(v, temperature, mass=1.0, kB=1.0, dim=2):
    """Equilibrium Maxwell-Boltzmann probability density of the speed :math:`|v|`.

    In :math:`d` dimensions,

    .. math::

        f(v) = v^{d-1}
               \\left(\\frac{m}{2\\pi k_B T}\\right)^{d/2}
               S_{d-1} \\, e^{-m v^2 / 2 k_B T}

    where :math:`S_{d-1}` is the surface area of the unit :math:`(d-1)`-sphere
    (:math:`2` for :math:`d=1`, :math:`2\\pi` for :math:`d=2`, :math:`4\\pi`
    for :math:`d=3`).

    Parameters
    ----------
    v : array_like
        Speed values (must be non-negative) at which to evaluate the density.
    temperature : float
        Temperature.
    mass : float, default=1.0
        Particle mass.
    kB : float, default=1.0
        Boltzmann constant.
    dim : {1, 2, 3}, default=2
        Spatial dimension.

    Returns
    -------
    ndarray
        Probability density :math:`f(v)`, normalized so that
        :math:`\\int_0^\\infty f(v)\\, dv = 1`.
    """
    v = np.asarray(v, dtype=np.float64)
    beta_m = mass / (kB * temperature)
    if dim == 1:
        surface = 2.0
    elif dim == 2:
        surface = 2.0 * np.pi
    elif dim == 3:
        surface = 4.0 * np.pi
    else:
        raise ValueError("dim must be 1, 2, or 3")
    prefactor = surface * (beta_m / (2.0 * np.pi)) ** (dim / 2.0)
    return prefactor * v ** (dim - 1) * np.exp(-0.5 * beta_m * v**2)


def maxwell_boltzmann_component_pdf(vx, temperature, mass=1.0, kB=1.0):
    """Equilibrium Gaussian density of a single Cartesian velocity component.

    .. math::

        f(v_x) = \\sqrt{\\frac{m}{2\\pi k_B T}} \\, e^{-m v_x^2 / 2 k_B T}

    Parameters
    ----------
    vx : array_like
        Velocity component values.
    temperature : float
        Temperature.
    mass : float, default=1.0
        Particle mass.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    ndarray
        Probability density :math:`f(v_x)`.
    """
    vx = np.asarray(vx, dtype=np.float64)
    beta_m = mass / (kB * temperature)
    return np.sqrt(beta_m / (2.0 * np.pi)) * np.exp(-0.5 * beta_m * vx**2)


def bose_einstein_occupation(energy, mu, temperature, kB=1.0):
    """Mean occupation number of a bosonic single-particle state, the Bose-Einstein distribution.

    .. math::

        n_{\\text{BE}}(\\varepsilon) = \\frac{1}{e^{(\\varepsilon - \\mu)/k_B T} - 1}

    Unlike Fermi-Dirac statistics, there is no restriction on how many
    identical bosons can occupy the same state; as the chemical potential
    :math:`\\mu` approaches the ground-state energy from below, the
    ground-state occupation diverges -- the onset of Bose-Einstein
    condensation.

    Parameters
    ----------
    energy : array_like
        Single-particle state energies. Must satisfy ``energy > mu``
        (otherwise the occupation is unphysical).
    mu : float
        Chemical potential.
    temperature : float
        Temperature.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    ndarray
        Mean occupation number :math:`n_{\\text{BE}}(\\varepsilon)`.
    """
    x = (np.asarray(energy, dtype=np.float64) - mu) / (kB * temperature)
    return 1.0 / np.expm1(x)


def fermi_dirac_occupation(energy, mu, temperature, kB=1.0):
    """Mean occupation number of a fermionic single-particle state, the Fermi-Dirac distribution.

    .. math::

        n_{\\text{FD}}(\\varepsilon) = \\frac{1}{e^{(\\varepsilon - \\mu)/k_B T} + 1}

    The Pauli exclusion principle caps :math:`n_{\\text{FD}}` at 1 for every
    state, in sharp contrast to Bose-Einstein statistics; as :math:`T \\to
    0`, this sharpens into a step function at the Fermi energy
    :math:`\\varepsilon = \\mu`.

    Parameters
    ----------
    energy : array_like
        Single-particle state energies.
    mu : float
        Chemical potential (the Fermi energy at :math:`T = 0`).
    temperature : float
        Temperature.
    kB : float, default=1.0
        Boltzmann constant.

    Returns
    -------
    ndarray
        Mean occupation number :math:`n_{\\text{FD}}(\\varepsilon) \\in (0, 1)`.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.all(fermi_dirac_occupation(np.array([0.0, 1.0, 2.0]), mu=1.0, temperature=0.01) >= 0))
    True
    """
    x = (np.asarray(energy, dtype=np.float64) - mu) / (kB * temperature)
    with np.errstate(over="ignore"):
        # exp(x) legitimately overflows to inf deep in the low-T step-function
        # limit; 1/(inf+1) still gives the mathematically correct 0 occupation.
        return 1.0 / (np.exp(x) + 1.0)


def bec_condensate_fraction(temperature, critical_temperature):
    """Condensate fraction of an ideal, homogeneous 3D Bose gas below the BEC transition.

    .. math::

        \\frac{N_0}{N} =
        \\begin{cases}
        1 - (T / T_c)^{3/2} & T < T_c \\\\
        0 & T \\ge T_c
        \\end{cases}

    A textbook prediction of Bose-Einstein statistics for an ideal gas in a
    box; in a 3D harmonic trap the exponent becomes 3 instead of 3/2.
    Condensation was first realized experimentally in 1995 in trapped
    dilute alkali-atom gases (Cornell and Wieman; Ketterle; 2001 Nobel Prize
    in Physics).

    Parameters
    ----------
    temperature : array_like
        Temperature(s) at which to evaluate the condensate fraction.
    critical_temperature : float
        The BEC transition temperature :math:`T_c`.

    Returns
    -------
    ndarray
        Condensate fraction :math:`N_0/N \\in [0, 1]`.

    Examples
    --------
    >>> f = bec_condensate_fraction(np.array([0.0, 0.5, 1.0, 2.0]), critical_temperature=1.0)
    >>> [round(float(v), 3) for v in f]
    [1.0, 0.646, 0.0, 0.0]
    """
    T = np.asarray(temperature, dtype=np.float64)
    fraction = 1.0 - (T / critical_temperature) ** 1.5
    return np.where(critical_temperature > T, fraction, 0.0)
