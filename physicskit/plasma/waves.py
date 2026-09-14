"""Cold plasma waves: the Stix dielectric tensor, dispersion relations, and CMA mapping.

Linearizes the multi-fluid cold-plasma equations around a uniform
background threaded by a uniform :math:`\\mathbf{B}_0` to get the
dielectric tensor components :math:`S`, :math:`D`, :math:`P` (Stix, 1962)
in :func:`stix_parameters`, then solves the general dispersion relation
:math:`\\mathbf{n}\\times(\\mathbf{n}\\times\\mathbf{E}) + \\mathbf{K}\\cdot\\mathbf{E}=0`
for the refractive index :math:`n=ck/\\omega` at any propagation angle
:math:`\\theta` to :math:`\\mathbf{B}_0` with :func:`cold_plasma_dispersion`.
Every named mode -- the parallel-propagating R- and L-waves, the
perpendicular O- and X-modes, and the low-frequency whistler branch -- is
a special case of that one quartic in :math:`n^2`.
:func:`cma_coordinates` maps a plasma state onto the two dimensionless
axes of the Clemmow-Mullaly-Allis diagram that organizes all of them.
"""

from __future__ import annotations

import numpy as np

from physicskit import constants as _const

__all__ = [
    "QE",
    "ME",
    "MP",
    "EPS0",
    "plasma_frequency",
    "stix_parameters",
    "rl_parameters",
    "cold_plasma_dispersion",
    "whistler_dispersion",
    "cma_coordinates",
    "ion_acoustic_soliton_profile",
    "ion_acoustic_soliton_evolve",
    "alfven_wave_pulse_ic",
    "simulate_alfven_wave",
]

QE = _const.ELEMENTARY_CHARGE
"""Elementary charge, in Coulombs."""

ME = _const.ELECTRON_MASS
"""Electron mass, in kilograms."""

MP = _const.PROTON_MASS
"""Proton mass, in kilograms."""

EPS0 = _const.VACUUM_PERMITTIVITY
"""Vacuum permittivity, in F/m."""


def plasma_frequency(n: float, q: float = QE, m: float = ME) -> float:
    """Species plasma frequency :math:`\\omega_p = \\sqrt{nq^2/(\\varepsilon_0 m)}`.

    The natural oscillation frequency of a species displaced from
    quasineutrality: the restoring electric field it builds up is
    proportional to the displacement, making every unmagnetized plasma a
    harmonic oscillator at this frequency -- the very phenomenon Langmuir
    identified in 1928.

    Parameters
    ----------
    n : float
        Number density in m^-3.
    q : float, default=QE
        Species charge magnitude in Coulombs.
    m : float, default=ME
        Species mass in kilograms.

    Returns
    -------
    float
        Angular plasma frequency in rad/s.

    Examples
    --------
    >>> round(float(plasma_frequency(1e19)) / 1e9, 3)
    178.399
    """
    return np.sqrt(n * q**2 / (EPS0 * m))


def stix_parameters(omega: float, B: float, species) -> tuple:
    """Stix cold-plasma dielectric tensor components :math:`S`, :math:`D`, :math:`P`.

    Summing each species' contribution to the linearized fluid response
    gives the Hermitian dielectric tensor

    .. math::

       \\mathbf{K} = \\begin{pmatrix} S & -iD & 0 \\\\ iD & S & 0 \\\\ 0 & 0 & P \\end{pmatrix},
       \\qquad
       S = 1 - \\sum_s \\frac{\\omega_{ps}^2}{\\omega^2-\\omega_{cs}^2}, \\quad
       D = \\sum_s \\frac{\\omega_{cs}}{\\omega}\\frac{\\omega_{ps}^2}{\\omega^2-\\omega_{cs}^2}, \\quad
       P = 1 - \\sum_s \\frac{\\omega_{ps}^2}{\\omega^2},

    with each species' signed cyclotron frequency :math:`\\omega_{cs}=q_sB/m_s`
    entering :math:`D` with its own sign -- the source of the
    circular-polarization asymmetry between the R- and L-waves.

    Parameters
    ----------
    omega : float
        Wave angular frequency in rad/s.
    B : float
        Background magnetic field magnitude in Tesla.
    species : sequence of (float, float, float)
        ``(n, q, m)`` for each plasma species: number density in m^-3,
        signed charge in Coulombs, mass in kilograms.

    Returns
    -------
    S, D, P : float
        Stix dielectric tensor components (dimensionless).

    See Also
    --------
    cold_plasma_dispersion : Solves the dispersion relation built from these.
    rl_parameters : The right/left-hand combinations :math:`R=S+D`, :math:`L=S-D`.

    Examples
    --------
    >>> electrons = (1e19, -QE, ME)
    >>> ions = (1e19, QE, MP)
    >>> S, D, P = stix_parameters(omega=2e9, B=1.0, species=[electrons, ions])
    >>> round(S, 4), round(D, 4), round(P, 4)
    (-2.3143, 90.6954, -7959.8516)
    """
    S, D, P = 1.0, 0.0, 1.0
    for n, q, m in species:
        wp2 = n * q**2 / (EPS0 * m)
        wc = q * B / m
        S -= wp2 / (omega**2 - wc**2)
        D += (wc / omega) * wp2 / (omega**2 - wc**2)
        P -= wp2 / omega**2
    return S, D, P


def rl_parameters(S: float, D: float) -> tuple:
    """Right- and left-hand Stix parameters :math:`R = S+D`, :math:`L = S-D`.

    :math:`R` and :math:`L` are the dielectric response seen by a purely
    right- or left-hand circularly polarized wave propagating exactly
    along :math:`\\mathbf{B}_0`; the R-wave resonates at the electron
    cyclotron frequency and the L-wave at the ion cyclotron frequency.

    Parameters
    ----------
    S, D : float
        Stix parameters from :func:`stix_parameters`.

    Returns
    -------
    R, L : float
        Right- and left-hand dielectric parameters.

    Examples
    --------
    >>> rl_parameters(S=1.0, D=0.5)
    (1.5, 0.5)
    """
    return S + D, S - D


def cold_plasma_dispersion(theta: float, S: float, D: float, P: float) -> tuple:
    """Solve the cold-plasma dispersion relation for the squared refractive index :math:`n^2`.

    Substituting a plane wave into
    :math:`\\mathbf{n}\\times(\\mathbf{n}\\times\\mathbf{E})+\\mathbf{K}\\cdot\\mathbf{E}=0`
    for propagation at angle :math:`\\theta` to :math:`\\mathbf{B}_0` gives
    the Appleton-Hartree biquadratic :math:`An^4 - Bn^2 + C = 0` with

    .. math::

       A = S\\sin^2\\theta + P\\cos^2\\theta, \\quad
       B = RL\\sin^2\\theta + PS(1+\\cos^2\\theta), \\quad
       C = PRL,

    which reduces at :math:`\\theta=0` to the decoupled R- and L-waves
    (:math:`n^2=R` or :math:`L`) and at :math:`\\theta=\\pi/2` to the
    O-mode (:math:`n^2=P`) and X-mode (:math:`n^2=RL/S`).

    Parameters
    ----------
    theta : float
        Angle between the wavevector and :math:`\\mathbf{B}_0`, in radians.
    S, D, P : float
        Stix parameters from :func:`stix_parameters`.

    Returns
    -------
    n_sq_plus, n_sq_minus : float
        The two roots of the biquadratic (the two cold-plasma wave
        branches at this angle and frequency). A negative root means that
        branch is evanescent rather than propagating.

    See Also
    --------
    stix_parameters : Supplies ``S``, ``D``, ``P``.

    Examples
    --------
    Parallel propagation recovers the pure R- and L-wave refractive indices:

    >>> import numpy as np
    >>> S, D, P = -2.314262935091991, 90.69535621969021, -7959.851640342367
    >>> n2_plus, n2_minus = cold_plasma_dispersion(theta=0.0, S=S, D=D, P=P)
    >>> R, L = rl_parameters(S, D)
    >>> bool(np.isclose(sorted([n2_plus, n2_minus]), sorted([R, L])).all())
    True
    """
    R, L = rl_parameters(S, D)
    A = S * np.sin(theta) ** 2 + P * np.cos(theta) ** 2
    Bc = R * L * np.sin(theta) ** 2 + P * S * (1.0 + np.cos(theta) ** 2)
    C = P * R * L
    disc = Bc**2 - 4.0 * A * C
    sqrt_disc = np.sqrt(disc)
    return (Bc + sqrt_disc) / (2.0 * A), (Bc - sqrt_disc) / (2.0 * A)


def whistler_dispersion(omega: float, wpe: float, wce: float) -> float:
    """Electron-only, parallel-propagation whistler-mode refractive index :math:`n^2 = \\omega_{pe}^2/[\\omega(\\omega_{ce}-\\omega)]`.

    The low-frequency (:math:`\\omega \\ll \\omega_{ce}`), right-hand
    circularly polarized branch of the R-wave, with ion motion neglected.
    Because :math:`n^2` grows without bound as :math:`\\omega\\to\\omega_{ce}`,
    whistlers slow down sharply as they approach the electron cyclotron
    frequency -- the falling-tone "whistle" heard in radio receivers after
    a lightning stroke launches a broadband pulse that disperses along
    Earth's field lines, with the highest frequencies (closest to
    :math:`\\omega_{ce}`) arriving last.

    Parameters
    ----------
    omega : float
        Wave angular frequency in rad/s, with :math:`\\omega \\ll \\omega_{ce}`.
    wpe : float
        Electron plasma frequency in rad/s, from :func:`plasma_frequency`.
    wce : float
        Electron cyclotron frequency in rad/s (positive).

    Returns
    -------
    float
        Squared refractive index :math:`n^2`.

    See Also
    --------
    cold_plasma_dispersion : The general dispersion relation this is a limit of.

    Examples
    --------
    Well below the electron cyclotron frequency, it agrees closely with
    the exact electron-only R-wave root of :func:`cold_plasma_dispersion`:

    >>> wpe, wce = 1.784e11, 1.759e11
    >>> omega = 0.001 * wce
    >>> S, D, P = stix_parameters(omega, B=1.0, species=[(wpe ** 2 * EPS0 * ME / QE ** 2, -QE, ME)])
    >>> R, _ = rl_parameters(S, D)
    >>> n2_approx = whistler_dispersion(omega, wpe, wce)
    >>> round(abs(n2_approx - R) / R, 2)
    0.0
    """
    return wpe**2 / (omega * (wce - omega))


def cma_coordinates(omega: float, wpe: float, wce: float) -> tuple:
    """Dimensionless :math:`(X, Y)` coordinates of the Clemmow-Mullaly-Allis (CMA) diagram.

    The CMA diagram partitions the plane :math:`X=\\omega_{pe}^2/\\omega^2`
    (density axis) vs. :math:`Y=\\omega_{ce}/\\omega` (field-strength axis)
    into regions of distinct wave topology -- cutoffs, resonances, and the
    number and polarization of propagating modes -- giving a single map
    of every cold-plasma wave regime from ordinary light waves
    (:math:`X, Y \\to 0`) to the Alfven wave and whistler branches
    (:math:`Y \\gg 1`).

    Parameters
    ----------
    omega : float
        Wave angular frequency in rad/s.
    wpe : float
        Electron plasma frequency in rad/s.
    wce : float
        Electron cyclotron frequency in rad/s.

    Returns
    -------
    X, Y : float
        CMA diagram coordinates.

    Examples
    --------
    >>> cma_coordinates(omega=1.0, wpe=2.0, wce=3.0)
    (4.0, 3.0)
    """
    return (wpe / omega) ** 2, wce / omega


def ion_acoustic_soliton_profile(x: np.ndarray, speed: float, x0: float = 0.0) -> np.ndarray:
    """Exact single-soliton solution of the ion-acoustic Korteweg-de Vries reduction.

    The reductive-perturbation (Washimi & Taniuti, 1966) expansion of the
    cold-ion-fluid/Boltzmann-electron equations in the weakly nonlinear,
    weakly dispersive limit reduces the ion-acoustic wave problem to the
    KdV equation for the normalized density (or potential) perturbation
    :math:`u` in a frame moving at the ion-sound speed; after the standard
    rescaling of the stretched coordinates it takes the canonical form
    :math:`u_t + 6uu_\\xi + u_{\\xi\\xi\\xi} = 0` -- the same normal form
    every weakly-dispersive weakly-nonlinear wave problem reduces to,
    with the ion-acoustic-specific physics (electron Boltzmann response
    supplying the nonlinearity, ion inertia and Debye-length dispersion
    supplying the :math:`\\partial_\\xi^3` term) fixing only the physical
    unit conversions between :math:`u,\\xi,t` and density, position, and
    time in the ion-sound-speed frame. Its exact traveling-wave solution
    is a single soliton of speed :math:`c` (in the stretched frame) and
    amplitude :math:`c/2`, propagating without change of shape --
    consistent with a real ion-acoustic soliton, whose speed always
    exceeds the linear sound speed by an amount set by its amplitude.

    Parameters
    ----------
    x : ndarray
        Spatial grid, in the stretched (ion-sound-speed) frame.
    speed : float
        Soliton speed :math:`c` in the stretched frame (amplitude
        :math:`=c/2`); must be positive.
    x0 : float, default=0.0
        Initial center position.

    Returns
    -------
    ndarray
        :math:`u(x, 0) = \\tfrac{c}{2}\\,\\mathrm{sech}^2\\!\\big(\\tfrac{\\sqrt{c}}{2}(x-x_0)\\big)`.

    See Also
    --------
    ion_acoustic_soliton_evolve : Propagate this (or any) initial condition forward in time.

    Examples
    --------
    >>> round(float(ion_acoustic_soliton_profile(0.0, speed=4.0)), 6)
    2.0
    """
    return (speed / 2.0) * (1.0 / np.cosh(np.sqrt(speed) / 2.0 * (x - x0))) ** 2


def _ion_acoustic_kdv_rhs_hat(u_hat: np.ndarray, k: np.ndarray) -> np.ndarray:
    u = np.real(np.fft.ifft(u_hat))
    ux = np.real(np.fft.ifft(1j * k * u_hat))
    return np.fft.fft(-6.0 * u * ux)


def ion_acoustic_soliton_evolve(u0: np.ndarray, x: np.ndarray, dt: float, steps: int) -> np.ndarray:
    """Evolve an ion-acoustic KdV initial condition forward in time on a periodic domain.

    A pseudo-spectral Strang-split scheme: the stiff linear dispersion
    :math:`\\partial_\\xi^3` is advanced exactly in Fourier space, and the
    non-stiff nonlinear advection :math:`6uu_\\xi` with RK4 in between --
    structurally the standard splitting for any KdV-type equation, applied
    here directly to the ion-acoustic reduction rather than by importing a
    generic KdV solver, since the physical field this equation governs
    (a Debye-length-normalized density/potential perturbation moving at
    order the ion-sound speed) is specific to this module.

    Parameters
    ----------
    u0 : ndarray
        Initial field, sampled on the periodic grid ``x``.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step.
    steps : int
        Number of steps to advance.

    Returns
    -------
    ndarray
        Field after ``steps * dt`` time units.

    See Also
    --------
    ion_acoustic_soliton_profile : Exact traveling-wave solution this reproduces.

    Examples
    --------
    A soliton launched at speed :math:`c=4` has advanced by very close to
    :math:`c\\cdot(\\text{steps}\\cdot dt)` and kept its amplitude, since
    KdV solitons propagate without changing shape:

    >>> import numpy as np
    >>> N, L = 512, 60.0
    >>> x = np.linspace(-L / 2, L / 2, N, endpoint=False)
    >>> u0 = ion_acoustic_soliton_profile(x, speed=4.0, x0=-15.0)
    >>> u = ion_acoustic_soliton_evolve(u0, x, dt=0.0005, steps=4000)
    >>> shift = x[np.argmax(u)] - x[np.argmax(u0)]
    >>> bool(abs(shift - 4.0 * 4000 * 0.0005) < 0.5)
    True
    >>> bool(abs(u.max() - u0.max()) < 0.05)
    True
    """
    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    ik3 = 1j * k**3
    u_hat = np.fft.fft(u0)
    for _ in range(steps):
        u_hat = u_hat * np.exp(ik3 * dt / 2.0)
        k1 = _ion_acoustic_kdv_rhs_hat(u_hat, k)
        k2 = _ion_acoustic_kdv_rhs_hat(u_hat + dt / 2.0 * k1, k)
        k3 = _ion_acoustic_kdv_rhs_hat(u_hat + dt / 2.0 * k2, k)
        k4 = _ion_acoustic_kdv_rhs_hat(u_hat + dt * k3, k)
        u_hat = u_hat + dt / 6.0 * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        u_hat = u_hat * np.exp(ik3 * dt / 2.0)
    return np.real(np.fft.ifft(u_hat))


def alfven_wave_pulse_ic(x: np.ndarray, x0: float, width: float, amplitude: float) -> tuple:
    """Initial condition for a transverse Alfven-wave pulse launched from rest.

    A Gaussian transverse-field pulse :math:`B_y(x,0)=B_1 e^{-[(x-x_0)/w]^2}`
    with the transverse velocity perturbation initially zero -- a "plucked
    string" initial condition. Because the linearized ideal-MHD Alfven-wave
    equations (:func:`simulate_alfven_wave`) are the same non-dispersive wave
    equation a string obeys, a disturbance released from rest splits exactly
    in half and propagates as two identical, oppositely directed pulses at
    :math:`\\pm v_A` -- exactly the field-line-plucking picture Alfven's
    original 1942 analogy describes.

    Parameters
    ----------
    x : ndarray
        Spatial grid (periodic).
    x0 : float
        Pulse center.
    width : float
        Gaussian pulse width.
    amplitude : float
        Peak transverse field perturbation :math:`B_1`.

    Returns
    -------
    By0, vy0 : ndarray
        Initial transverse magnetic field and velocity perturbations, same
        shape as `x` (`vy0` identically zero).

    See Also
    --------
    simulate_alfven_wave : Evolves this initial condition forward in time.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-10, 10, 64, endpoint=False)
    >>> By0, vy0 = alfven_wave_pulse_ic(x, x0=0.0, width=1.0, amplitude=0.1)
    >>> bool(np.all(vy0 == 0.0))
    True
    """
    By0 = amplitude * np.exp(-(((x - x0) / width) ** 2))
    vy0 = np.zeros_like(x)
    return By0, vy0


def _alfven_rhs(By: np.ndarray, vy: np.ndarray, k: np.ndarray, B0: float, rho0: float, mu0: float) -> tuple:
    dBy_dx = np.real(np.fft.ifft(1j * k * np.fft.fft(By)))
    dvy_dx = np.real(np.fft.ifft(1j * k * np.fft.fft(vy)))
    dBy_dt = B0 * dvy_dx
    dvy_dt = (B0 / (mu0 * rho0)) * dBy_dx
    return dBy_dt, dvy_dt


def simulate_alfven_wave(By0: np.ndarray, vy0: np.ndarray, x: np.ndarray, dt: float, steps: int, B0: float, rho0: float, mu0: float = 1.0) -> dict:
    """Time-step the linearized 1D ideal-MHD Alfven-wave equations with a pseudo-spectral RK4 scheme.

    Advances the coupled transverse induction and momentum equations

    .. math::

       \\partial_t B_y = B_0\\,\\partial_x v_y, \\qquad
       \\rho_0\\,\\partial_t v_y = \\frac{B_0}{\\mu_0}\\,\\partial_x B_y,

    which combine into the non-dispersive wave equation
    :math:`\\partial_t^2 B_y = v_A^2\\,\\partial_x^2 B_y` with
    :math:`v_A=B_0/\\sqrt{\\mu_0\\rho_0}` (:func:`physicskit.plasma.mhd.alfven_speed`).
    Spatial derivatives are evaluated exactly via FFT (as elsewhere in this
    package's periodic-domain solvers) and advanced in time with classical
    RK4; a uniform background field and density are assumed throughout
    (linear, ideal, cold-background MHD -- no thermal pressure term enters
    a purely transverse, incompressible perturbation like this one).

    Unlike :func:`physicskit.plasma.mhd.alfven_speed` (SI units), this
    solver defaults to normalized units (``mu0=1.0``, matching order-unity
    `B0`/`rho0`) since with the true SI :math:`\\mu_0\\approx1.26\\times10^{-6}`
    a Tesla-scale field gives an Alfven speed of order :math:`10^5`-:math:`10^6`
    m/s, requiring correspondingly tiny time steps to satisfy the CFL bound
    below; pass ``mu0=physicskit.plasma.mhd.MU0`` explicitly for SI-consistent
    parameters and scale `dt` accordingly.

    Parameters
    ----------
    By0, vy0 : ndarray
        Initial transverse field and velocity perturbations, e.g. from
        :func:`alfven_wave_pulse_ic`.
    x : ndarray
        Uniformly spaced periodic spatial grid.
    dt : float
        Time step; the CFL condition :math:`v_A\\,dt \\le dx` should be
        respected for the explicit RK4 stepping to remain stable.
    steps : int
        Number of RK4 steps to advance.
    B0 : float
        Background field magnitude.
    rho0 : float
        Background mass density.
    mu0 : float, default=1.0
        Vacuum permeability (normalized units by default; see above).

    Returns
    -------
    dict
        ``{"By": final transverse field, "vy": final transverse velocity}``.

    See Also
    --------
    alfven_wave_pulse_ic : Builds the initial condition consumed here.
    physicskit.plasma.mhd.alfven_speed : The propagation speed this recovers.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-20, 20, 256, endpoint=False)
    >>> By0, vy0 = alfven_wave_pulse_ic(x, x0=0.0, width=1.0, amplitude=0.1)
    >>> result = simulate_alfven_wave(By0, vy0, x, dt=0.002, steps=2000, B0=1.0, rho0=1.0)
    >>> result["By"].shape
    (256,)
    >>> bool(np.isfinite(result["By"]).all())
    True
    """
    k = 2.0 * np.pi * np.fft.fftfreq(len(x), d=x[1] - x[0])
    By = np.array(By0, dtype=float, copy=True)
    vy = np.array(vy0, dtype=float, copy=True)
    for _ in range(steps):
        k1B, k1v = _alfven_rhs(By, vy, k, B0, rho0, mu0)
        k2B, k2v = _alfven_rhs(By + dt / 2 * k1B, vy + dt / 2 * k1v, k, B0, rho0, mu0)
        k3B, k3v = _alfven_rhs(By + dt / 2 * k2B, vy + dt / 2 * k2v, k, B0, rho0, mu0)
        k4B, k4v = _alfven_rhs(By + dt * k3B, vy + dt * k3v, k, B0, rho0, mu0)
        By = By + dt / 6 * (k1B + 2 * k2B + 2 * k3B + k4B)
        vy = vy + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return {"By": By, "vy": vy}
