"""Magnetohydrodynamics: wave speeds, toroidal equilibrium, and magnetic reconnection.

Treats the plasma as a single conducting fluid rather than a collection of
orbiting particles. Three regimes are covered: the characteristic wave
speeds of ideal MHD (:func:`alfven_speed`, :func:`sound_speed`,
:func:`magnetosonic_speeds`); static toroidal equilibrium, obtained by
solving the Grad-Shafranov equation
:math:`\\Delta^*\\psi = -\\mu_0 R^2 p'(\\psi) - FF'(\\psi)` for the poloidal
flux function :math:`\\psi(R, Z)` that balances pressure against the
magnetic force in an axisymmetric device (:func:`solve_grad_shafranov`,
:func:`safety_factor_large_aspect_ratio`); and resistive magnetic
reconnection, where a thin current sheet lets field lines of opposite
polarity break and reconnect, converting magnetic energy to heat and flow
(:func:`sweet_parker_rate`, :func:`petschek_rate`).
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "MU0",
    "alfven_speed",
    "sound_speed",
    "magnetosonic_speeds",
    "solovev_particular_solution",
    "grad_shafranov_rhs_solovev",
    "solve_grad_shafranov",
    "safety_factor_large_aspect_ratio",
    "lundquist_number",
    "resistive_diffusion_time",
    "sweet_parker_rate",
    "sweet_parker_layer_width",
    "petschek_rate",
]

MU0 = 4 * np.pi * 1e-7
"""Vacuum permeability, in H/m."""


def alfven_speed(B: float, rho: float) -> float:
    """Alfven speed :math:`v_A = B/\\sqrt{\\mu_0 \\rho}`, at which a perturbation propagates along tensioned field lines.

    Hannes Alfven's 1942 discovery that a magnetized, perfectly conducting
    fluid supports a transverse wave -- field lines behaving like strings
    under tension :math:`B^2/\\mu_0`, plucked by the inertia of the frozen-in
    plasma -- founded MHD as a distinct discipline and earned the 1970
    Nobel Prize.

    Parameters
    ----------
    B : float
        Magnetic field magnitude in Tesla.
    rho : float
        Mass density in kg/m^3.

    Returns
    -------
    float
        Alfven speed in m/s.

    See Also
    --------
    magnetosonic_speeds : The compressive (fast/slow) counterparts of this
        purely magnetic wave.

    Examples
    --------
    >>> round(float(alfven_speed(B=1.0, rho=1e-6)), 2)
    892062.06
    """
    return B / np.sqrt(MU0 * rho)


def sound_speed(gamma: float, p: float, rho: float) -> float:
    """Ordinary adiabatic sound speed :math:`c_s = \\sqrt{\\gamma p/\\rho}`.

    Parameters
    ----------
    gamma : float
        Adiabatic index (5/3 for an ideal monatomic gas).
    p : float
        Pressure in Pa.
    rho : float
        Mass density in kg/m^3.

    Returns
    -------
    float
        Sound speed in m/s.

    Examples
    --------
    >>> round(float(sound_speed(gamma=5 / 3, p=1.0, rho=1e-6)), 2)
    1290.99
    """
    return np.sqrt(gamma * p / rho)


def magnetosonic_speeds(vA: float, cs: float, theta: float) -> tuple:
    """Fast and slow magnetosonic phase speeds at propagation angle :math:`\\theta` to :math:`\\mathbf{B}`.

    The two compressive MHD normal modes solve
    :math:`v_{f,s}^2 = \\tfrac{1}{2}\\left[(v_A^2+c_s^2) \\pm \\sqrt{(v_A^2+c_s^2)^2 - 4v_A^2c_s^2\\cos^2\\theta}\\right]`.
    At :math:`\\theta=0` (propagation along :math:`\\mathbf{B}`) they reduce
    to :math:`\\max(v_A, c_s)` and :math:`\\min(v_A, c_s)`; at
    :math:`\\theta=\\pi/2` the fast mode becomes the purely compressive
    :math:`\\sqrt{v_A^2+c_s^2}` and the slow mode vanishes, since a
    perpendicular perturbation cannot bend field lines that are already
    perpendicular to its wavevector.

    Parameters
    ----------
    vA : float
        Alfven speed in m/s, from :func:`alfven_speed`.
    cs : float
        Sound speed in m/s, from :func:`sound_speed`.
    theta : float
        Angle between the wavevector and :math:`\\mathbf{B}`, in radians.

    Returns
    -------
    v_fast, v_slow : float
        Fast and slow magnetosonic phase speeds in m/s.

    Examples
    --------
    >>> import numpy as np
    >>> vf, vs = magnetosonic_speeds(vA=892062.06, cs=1e5, theta=np.pi / 2)
    >>> round(vf, 2)
    897649.55
    >>> round(vs, 2)
    0.0
    """
    s = vA**2 + cs**2
    d = np.sqrt(max(s**2 - 4.0 * vA**2 * cs**2 * np.cos(theta) ** 2, 0.0))
    return float(np.sqrt(0.5 * (s + d))), float(np.sqrt(0.5 * (s - d)))


def solovev_particular_solution(R: np.ndarray, Z: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Closed-form particular solution of the linear (Solov'ev) Grad-Shafranov equation.

    When the source term is linear in :math:`R^2` -- i.e.
    :math:`p'(\\psi) = \\text{const}` and :math:`FF'(\\psi) = \\text{const}`,
    so :math:`\\Delta^*\\psi = c_1 R^2 + c_2` -- the Grad-Shafranov operator
    :math:`\\Delta^*\\psi = \\partial_R^2\\psi - R^{-1}\\partial_R\\psi + \\partial_Z^2\\psi`
    admits the exact polynomial solution :math:`\\psi_p = \\tfrac{c_1}{8}R^4 +
    \\tfrac{c_2}{2}Z^2` (Solov'ev, 1968); adding any solution of the
    homogeneous equation :math:`\\Delta^*\\psi=0` shapes the boundary into a
    D-shaped or elongated cross-section without affecting the pressure and
    current profile. Used here to validate :func:`solve_grad_shafranov`
    against an exact answer, and to supply consistent Dirichlet boundary
    data for it.

    Parameters
    ----------
    R, Z : ndarray
        Cylindrical coordinates (broadcastable), in meters.
    c1, c2 : float
        Coefficients of the linear source term :math:`\\Delta^*\\psi = c_1 R^2 + c_2`.

    Returns
    -------
    ndarray
        Poloidal flux :math:`\\psi(R, Z)`, same shape as ``R``/``Z``.

    See Also
    --------
    grad_shafranov_rhs_solovev : The corresponding source term.
    solve_grad_shafranov : Numerical solver validated against this solution.

    Examples
    --------
    >>> import numpy as np
    >>> solovev_particular_solution(np.array([1.0]), np.array([0.0]), c1=1.0, c2=-2.0)
    array([0.125])
    """
    return (c1 / 8.0) * R**4 + (c2 / 2.0) * Z**2


def grad_shafranov_rhs_solovev(R: np.ndarray, c1: float, c2: float) -> np.ndarray:
    """Source term :math:`\\Delta^*\\psi = c_1 R^2 + c_2` of the linear Solov'ev equilibrium.

    Equivalent to :math:`-\\mu_0 R^2 p'(\\psi) - FF'(\\psi)` in the
    Grad-Shafranov equation for the special case of constant
    :math:`p'(\\psi)` and :math:`FF'(\\psi)`.

    Parameters
    ----------
    R : ndarray
        Major-radius coordinate, in meters.
    c1, c2 : float
        Source coefficients (related to :math:`p'` and :math:`FF'`).

    Returns
    -------
    ndarray
        The right-hand side :math:`\\Delta^*\\psi`, same shape as ``R``.

    Examples
    --------
    >>> import numpy as np
    >>> grad_shafranov_rhs_solovev(np.array([1.0, 2.0]), c1=1.0, c2=-2.0)
    array([-1.,  2.])
    """
    return c1 * R**2 + c2


@njit(cache=True)
def _sor_grad_shafranov(psi, R, dR, dZ, c1, c2, omega, max_iter):
    nr, nz = psi.shape
    for _ in range(max_iter):
        for i in range(1, nr - 1):
            Ri = R[i]
            coef_ip = 1.0 / dR**2 - 1.0 / (2.0 * Ri * dR)
            coef_im = 1.0 / dR**2 + 1.0 / (2.0 * Ri * dR)
            coef_j = 1.0 / dZ**2
            coef_c = -2.0 / dR**2 - 2.0 / dZ**2
            for j in range(1, nz - 1):
                source = c1 * Ri**2 + c2
                rhs = source - coef_ip * psi[i + 1, j] - coef_im * psi[i - 1, j] - coef_j * psi[i, j + 1] - coef_j * psi[i, j - 1]
                psi[i, j] = (1.0 - omega) * psi[i, j] + omega * (rhs / coef_c)
    return psi


def solve_grad_shafranov(R: np.ndarray, Z: np.ndarray, c1: float, c2: float, omega: float = 1.8, max_iter: int = 4000) -> np.ndarray:
    """Solve the axisymmetric Grad-Shafranov equation by successive over-relaxation (SOR).

    Finite-differences the elliptic operator
    :math:`\\Delta^*\\psi = \\partial_R^2\\psi - R^{-1}\\partial_R\\psi +
    \\partial_Z^2\\psi` on a rectangular :math:`(R, Z)` grid and relaxes it
    toward the linear (Solov'ev) source :math:`c_1 R^2 + c_2`, using
    Dirichlet boundary data taken from the exact
    :func:`solovev_particular_solution` -- so the interior solution this
    converges to is known analytically and can be checked directly, rather
    than only visually.

    Parameters
    ----------
    R : ndarray, shape (nr,)
        Major-radius grid points, in meters; must be strictly positive
        (the operator is singular at :math:`R=0`).
    Z : ndarray, shape (nz,)
        Vertical grid points, in meters.
    c1, c2 : float
        Coefficients of the linear source term, as in :func:`grad_shafranov_rhs_solovev`.
    omega : float, default=1.8
        SOR relaxation parameter, :math:`1 < \\omega < 2`.
    max_iter : int, default=4000
        Number of relaxation sweeps.

    Returns
    -------
    ndarray, shape (nr, nz)
        Poloidal flux :math:`\\psi(R, Z)` on the grid.

    See Also
    --------
    solovev_particular_solution : The exact solution this converges to.
    safety_factor_large_aspect_ratio : A downstream equilibrium diagnostic.

    Examples
    --------
    >>> import numpy as np
    >>> R = np.linspace(0.5, 1.5, 41)
    >>> Z = np.linspace(-0.5, 0.5, 41)
    >>> c1, c2 = 1.0, -2.0
    >>> psi = solve_grad_shafranov(R, Z, c1, c2)
    >>> RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    >>> psi_exact = solovev_particular_solution(RR, ZZ, c1, c2)
    >>> bool(np.max(np.abs(psi - psi_exact)) < 1e-3)
    True
    """
    nr, nz = len(R), len(Z)
    RR, ZZ = np.meshgrid(R, Z, indexing="ij")
    psi_exact = solovev_particular_solution(RR, ZZ, c1, c2)
    psi = np.zeros((nr, nz))
    psi[0, :] = psi_exact[0, :]
    psi[-1, :] = psi_exact[-1, :]
    psi[:, 0] = psi_exact[:, 0]
    psi[:, -1] = psi_exact[:, -1]
    dR = float(R[1] - R[0])
    dZ = float(Z[1] - Z[0])
    return _sor_grad_shafranov(psi, np.asarray(R, dtype=float), dR, dZ, float(c1), float(c2), float(omega), int(max_iter))


def safety_factor_large_aspect_ratio(r: float, R0: float, Bt: float, Bp: float) -> float:
    """Tokamak safety factor :math:`q \\approx rB_t/(R_0 B_p)` in the large-aspect-ratio approximation.

    Counts how many times a field line winds the long way (toroidally)
    around the torus for each time it winds the short way (poloidally).
    Field lines with rational :math:`q = m/n` close on themselves after
    :math:`n` toroidal transits and are resonant surfaces for
    magnetic-island-forming instabilities; :math:`q=1` in particular marks
    the sawtooth-unstable region at a tokamak's core.

    Parameters
    ----------
    r : float
        Minor-radius coordinate of the flux surface, in meters.
    R0 : float
        Major radius of the torus, in meters.
    Bt : float
        Toroidal field strength at the flux surface, in Tesla.
    Bp : float
        Poloidal field strength at the flux surface, in Tesla.

    Returns
    -------
    float
        Safety factor (dimensionless).

    Examples
    --------
    >>> round(safety_factor_large_aspect_ratio(r=0.3, R0=1.0, Bt=2.0, Bp=0.2), 10)
    3.0
    """
    return r * Bt / (R0 * Bp)


def lundquist_number(L: float, vA: float, eta: float) -> float:
    """Lundquist number :math:`S = L v_A/\\eta`, the magnetic Reynolds number built from the Alfven speed.

    Parameters
    ----------
    L : float
        Characteristic length scale (e.g. current-sheet length), in meters.
    vA : float
        Alfven speed in m/s, from :func:`alfven_speed`.
    eta : float
        Magnetic diffusivity in m^2/s.

    Returns
    -------
    float
        Lundquist number (dimensionless). Fusion and astrophysical plasmas
        typically have :math:`S \\sim 10^{6}` -- :math:`10^{14}`.

    See Also
    --------
    sweet_parker_rate : Reconnection rate scaling as :math:`S^{-1/2}`.

    Examples
    --------
    >>> lundquist_number(L=1.0, vA=1e6, eta=1.0)
    1000000.0
    """
    return L * vA / eta


def resistive_diffusion_time(L: float, eta: float) -> float:
    """Resistive diffusion time :math:`\\tau_\\eta = L^2/\\eta` for magnetic field to decay through a length :math:`L`.

    Parameters
    ----------
    L : float
        Length scale in meters.
    eta : float
        Magnetic diffusivity in m^2/s.

    Returns
    -------
    float
        Diffusion time in seconds.

    Examples
    --------
    >>> resistive_diffusion_time(L=1.0, eta=1.0)
    1.0
    """
    return L**2 / eta


def sweet_parker_rate(S: float) -> float:
    """Sweet-Parker reconnection rate :math:`v_{in}/v_A = S^{-1/2}`.

    Sweet and Parker's 1957/1958 model treats reconnection as
    steady inflow through a long, thin resistive current sheet of aspect
    ratio :math:`\\delta/L \\sim S^{-1/2}`; mass conservation through that
    narrow sheet throttles the inflow (and hence the whole reconnection
    process) to the same :math:`S^{-1/2}` scaling. For solar-flare-scale
    Lundquist numbers (:math:`S\\sim10^{12}`) this predicts reconnection
    millions of times too slow to explain observed flare energy-release
    times -- the puzzle :func:`petschek_rate` was proposed to resolve.

    Parameters
    ----------
    S : float
        Lundquist number, from :func:`lundquist_number`.

    Returns
    -------
    float
        Dimensionless reconnection rate :math:`v_{in}/v_A`.

    See Also
    --------
    sweet_parker_layer_width : The current-sheet thickness behind this rate.
    petschek_rate : The faster, X-point reconnection alternative.

    Examples
    --------
    >>> round(float(sweet_parker_rate(S=1e6)), 6)
    0.001
    """
    return 1.0 / np.sqrt(S)


def sweet_parker_layer_width(L: float, S: float) -> float:
    """Sweet-Parker current-sheet thickness :math:`\\delta = L/\\sqrt{S}`.

    Parameters
    ----------
    L : float
        Current-sheet length in meters.
    S : float
        Lundquist number, from :func:`lundquist_number`.

    Returns
    -------
    float
        Current-sheet thickness in meters.

    Examples
    --------
    >>> round(float(sweet_parker_layer_width(L=1e7, S=1e6)), 4)
    10000.0
    """
    return L / np.sqrt(S)


def petschek_rate(S: float) -> float:
    """Petschek reconnection rate :math:`v_{in}/v_A \\approx \\pi/(8\\ln S)`.

    Petschek (1964) showed that if the diffusion region shrinks to a
    small X-point rather than the full Sweet-Parker sheet length, four
    standing slow-mode shocks can carry most of the inflowing flux and
    energy conversion, giving a reconnection rate that falls only
    logarithmically with :math:`S` instead of as :math:`S^{-1/2}` --
    fast enough to plausibly explain solar flare and magnetospheric
    substorm timescales.

    Parameters
    ----------
    S : float
        Lundquist number, from :func:`lundquist_number`.

    Returns
    -------
    float
        Dimensionless reconnection rate :math:`v_{in}/v_A`.

    See Also
    --------
    sweet_parker_rate : The slower, steady-sheet reconnection rate this improves on.

    Examples
    --------
    >>> round(float(petschek_rate(S=1e6)), 4)
    0.0284
    """
    return np.pi / (8.0 * np.log(S))
