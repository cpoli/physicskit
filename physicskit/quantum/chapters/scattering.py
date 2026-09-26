r"""Potential scattering in three dimensions: the Born approximation and partial waves.

Units :math:`\hbar = 1`. A particle of mass :math:`m` and wavenumber
:math:`k` scatters off a central potential :math:`V(r)`; the scattering
amplitude :math:`f(\theta)` gives the differential cross section
:math:`d\sigma/d\Omega = |f(\theta)|^2`.

- :func:`momentum_transfer`, :func:`born_amplitude`,
  :func:`yukawa_born_amplitude` -- the first Born approximation (Born, Z.
  Phys. 38, 803 (1926)), numerically for any central potential and in
  closed form for the Yukawa/screened-Coulomb potential (whose unscreened
  limit is Rutherford's formula).
- :func:`partial_wave_phase_shifts`, :func:`born_phase_shifts`,
  :func:`hard_sphere_phase_shifts` -- phase shifts :math:`\delta_\ell(k)`
  from the variable-phase equation, the Born approximation, and the exact
  hard-sphere result.
- :func:`partial_wave_amplitude`, :func:`partial_wave_cross_section` --
  resumming the partial-wave series.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import numpy as np
from scipy.integrate import quad, solve_ivp
from scipy.special import eval_legendre, spherical_jn, spherical_yn

__all__ = [
    "momentum_transfer",
    "born_amplitude",
    "yukawa_born_amplitude",
    "partial_wave_phase_shifts",
    "born_phase_shifts",
    "hard_sphere_phase_shifts",
    "partial_wave_amplitude",
    "partial_wave_cross_section",
]


def momentum_transfer(k: float, theta: np.ndarray) -> np.ndarray:
    r"""Momentum transfer :math:`q = |\mathbf k' - \mathbf k| = 2k\sin(\theta/2)` for elastic scattering.

    Parameters
    ----------
    k : float
        Wavenumber.
    theta : array_like
        Scattering angle (radians).

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> float(momentum_transfer(1.5, np.pi))
    3.0
    """
    return 2 * k * np.sin(np.asarray(theta, dtype=float) / 2)


def born_amplitude(potential: Callable[[np.ndarray], np.ndarray], q: np.ndarray, mass: float = 1.0) -> np.ndarray:
    r"""First Born approximation to the scattering amplitude of a central potential.

    .. math::

        f_B(q) = -\frac{m}{2\pi}\int d^3r\, e^{-i\mathbf q\cdot\mathbf r} V(r)
               = -\frac{2m}{q}\int_0^\infty r\,V(r)\sin(qr)\,dr

    (Born 1926; Sakurai, *Modern Quantum Mechanics*, Eq. 7.2.10), with
    :math:`q \to 0` limit :math:`-2m\int_0^\infty r^2 V\,dr`. The
    oscillatory integral is done with QUADPACK's Fourier-integral routine
    (:func:`scipy.integrate.quad` with ``weight="sin"``), which needs
    :math:`rV(r)` to decay at large :math:`r`.

    Parameters
    ----------
    potential : callable
        :math:`V(r)`, vectorized over ``r``.
    q : array_like
        Momentum transfer(s), :math:`q \ge 0`.
    mass : float, default=1.0
        Particle (reduced) mass.

    Returns
    -------
    numpy.ndarray
        Real Born amplitude :math:`f_B(q)`.

    Examples
    --------
    A Gaussian well :math:`V = V_0 e^{-r^2}` has
    :math:`f_B = -\tfrac{\sqrt\pi}{2} m V_0\,e^{-q^2/4}`:

    >>> import numpy as np
    >>> f = born_amplitude(lambda r: -0.3 * np.exp(-r**2), 1.2)
    >>> bool(np.isclose(f, 0.5 * np.sqrt(np.pi) * 0.3 * np.exp(-0.36)))
    True
    """

    def r_pow_V(r, power):
        # the quadrature may sample r = 0, where e.g. a Coulomb-like 1/r is
        # singular; that single point carries no weight in the integral
        with np.errstate(divide="ignore", invalid="ignore"):
            val = float(r**power * potential(r))
        return val if np.isfinite(val) else 0.0

    q_arr = np.atleast_1d(np.asarray(q, dtype=float))
    out = np.empty_like(q_arr)
    for i, qi in enumerate(q_arr):
        if qi < 1e-12:
            out[i] = -2 * mass * quad(r_pow_V, 0, np.inf, args=(2,), limit=500)[0]
        else:
            integral = quad(r_pow_V, 0, np.inf, args=(1,), weight="sin", wvar=qi, limlst=200)[0]
            out[i] = -2 * mass * integral / qi
    return out if np.ndim(q) else out[0]


def yukawa_born_amplitude(q: np.ndarray, g: float, mu: float, mass: float = 1.0) -> np.ndarray:
    r"""Closed-form Born amplitude of the Yukawa potential :math:`V(r) = g\,e^{-\mu r}/r`.

    .. math::

        f_B(q) = -\frac{2mg}{\mu^2 + q^2}

    (Sakurai, Eq. 7.2.15). As :math:`\mu \to 0` (unscreened Coulomb,
    :math:`g = Z_1Z_2\alpha`) :math:`|f_B|^2` becomes Rutherford's
    classical cross section,

    .. math::

        \frac{d\sigma}{d\Omega} = \frac{4m^2g^2}{q^4}
            = \left(\frac{g}{4E}\right)^2 \frac{1}{\sin^4(\theta/2)},
        \qquad E = \frac{k^2}{2m}.

    Parameters
    ----------
    q : array_like
        Momentum transfer.
    g : float
        Coupling strength (:math:`g > 0` repulsive).
    mu : float
        Inverse screening length.
    mass : float, default=1.0
        Particle (reduced) mass.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> float(yukawa_born_amplitude(1.0, g=0.5, mu=1.0))
    -0.5
    """
    q = np.asarray(q, dtype=float)
    return -2 * mass * g / (mu**2 + q**2)


def partial_wave_phase_shifts(
    potential: Callable[[np.ndarray], np.ndarray],
    k: float,
    l_max: int,
    mass: float = 1.0,
    r_max: float = 30.0,
    r_min: float = 1e-8,
    breakpoints: Sequence[float] = (),
    rtol: float = 1e-10,
) -> np.ndarray:
    r"""Phase shifts :math:`\delta_\ell(k)`, :math:`\ell = 0,\dots,\ell_{\max}`, of a short-range central potential.

    Integrates Calogero's variable-phase equation

    .. math::

        \frac{d\delta_\ell}{dr} = -\frac{U(r)}{k}\left[\hat\jmath_\ell(kr)\cos\delta_\ell(r)
            - \hat n_\ell(kr)\sin\delta_\ell(r)\right]^2,
        \qquad U = 2mV,

    from :math:`\delta_\ell(0) = 0` out to ``r_max``, with the
    Riccati-Bessel functions :math:`\hat\jmath_\ell(x) = x j_\ell(x)` and
    :math:`\hat n_\ell(x) = x y_\ell(x)`. :math:`\delta_\ell(r)` is the
    phase shift produced by the potential truncated at :math:`r`, so its
    limit is the full phase shift (F. Calogero, *Variable Phase Approach
    to Potential Scattering*, Academic Press, 1967, Eq. 3.4). Unlike
    shooting and matching, it never overflows at large :math:`\ell` and
    gives a continuous :math:`\delta_\ell` (not reduced modulo
    :math:`\pi`).

    Parameters
    ----------
    potential : callable
        :math:`V(r)`, finite for :math:`r > 0` and negligible beyond
        ``r_max``.
    k : float
        Wavenumber, :math:`k > 0`.
    l_max : int
        Highest partial wave.
    mass : float, default=1.0
        Particle (reduced) mass.
    r_max : float, default=30.0
        Outer integration radius.
    r_min : float, default=1e-8
        Inner starting radius.
    breakpoints : sequence of float, optional
        Radii where :math:`V` is discontinuous (e.g. a square well's
        edge). The integration restarts there instead of shrinking its
        step size to resolve the jump.
    rtol : float, default=1e-10
        Relative tolerance of the ODE solver (absolute tolerance is
        ``rtol / 100``).

    Returns
    -------
    numpy.ndarray
        Shape ``(l_max + 1,)``, in radians.

    Examples
    --------
    An attractive square well of depth :math:`V_0` and radius :math:`a`
    has :math:`\tan(ka + \delta_0) = (k/K)\tan(Ka)`,
    :math:`K = \sqrt{k^2 + 2mV_0}`:

    >>> import numpy as np
    >>> well = lambda r: np.where(r < 1.0, -0.8, 0.0)
    >>> d0 = partial_wave_phase_shifts(well, k=0.7, l_max=0, r_max=1.5, breakpoints=[1.0])[0]
    >>> K = np.sqrt(0.49 + 1.6)
    >>> bool(np.isclose(np.tan(0.7 + d0), 0.7 / K * np.tan(K)))
    True
    """
    ells = np.arange(l_max + 1)

    def rhs(r, delta):
        x = k * r
        jh = x * spherical_jn(ells, x)
        nh = x * spherical_yn(ells, x)
        return -2 * mass * potential(r) / k * (jh * np.cos(delta) - nh * np.sin(delta)) ** 2

    edges = [r_min, *sorted(b for b in breakpoints if r_min < b < r_max), r_max]
    delta = np.zeros(l_max + 1)
    for r0, r1 in zip(edges[:-1], edges[1:]):
        # evaluate V strictly inside each segment so a jump at r1 is never sampled
        sol = solve_ivp(rhs, (r0, r1 * (1 - 1e-15)), delta, method="DOP853", rtol=rtol, atol=rtol / 100)
        delta = sol.y[:, -1]
    return delta


def born_phase_shifts(
    potential: Callable[[np.ndarray], np.ndarray],
    k: float,
    l_max: int,
    mass: float = 1.0,
    r_max: float = 30.0,
) -> np.ndarray:
    r"""Phase shifts in the Born approximation, :math:`\delta_\ell \approx -2mk\int_0^\infty r^2 V(r)\,j_\ell(kr)^2\,dr`.

    The small-:math:`V` limit of the variable-phase equation (Sakurai,
    Eq. 7.6.22); valid when every :math:`|\delta_\ell| \ll 1`.

    Parameters
    ----------
    potential : callable
        :math:`V(r)`.
    k : float
        Wavenumber.
    l_max : int
        Highest partial wave.
    mass : float, default=1.0
        Particle (reduced) mass.
    r_max : float, default=30.0
        Outer integration radius.

    Returns
    -------
    numpy.ndarray
        Shape ``(l_max + 1,)``.

    Examples
    --------
    >>> import numpy as np
    >>> weak = lambda r: -0.01 * np.exp(-r**2)
    >>> d_born = born_phase_shifts(weak, k=1.0, l_max=2)
    >>> d_exact = partial_wave_phase_shifts(weak, k=1.0, l_max=2, r_max=8.0)
    >>> bool(np.allclose(d_born, d_exact, rtol=0.02))
    True
    """
    return np.array([-2 * mass * k * quad(lambda r: r**2 * potential(r) * spherical_jn(ell, k * r) ** 2, 0, r_max, limit=500)[0] for ell in range(l_max + 1)])


def hard_sphere_phase_shifts(k: float, a: float, l_max: int) -> np.ndarray:
    r"""Exact phase shifts of an impenetrable sphere of radius :math:`a`: :math:`\tan\delta_\ell = j_\ell(ka)/y_\ell(ka)`.

    The radial wavefunction :math:`\propto \cos\delta_\ell\,j_\ell(kr) -
    \sin\delta_\ell\,y_\ell(kr)` must vanish at :math:`r = a` (Sakurai,
    Eq. 7.6.31); in particular :math:`\delta_0 = -ka` (modulo :math:`\pi`). The total cross
    section tends to :math:`4\pi a^2` as :math:`ka \to 0` and to
    :math:`2\pi a^2` as :math:`ka \to \infty` (the extra :math:`\pi a^2` is
    the diffraction "shadow"), approached as
    :math:`2\pi a^2[1 + 0.9962\,(ka)^{-2/3}]` (Nussenzveig, J. Math. Phys.
    10, 82 (1969)).

    Parameters
    ----------
    k : float
        Wavenumber.
    a : float
        Sphere radius.
    l_max : int
        Highest partial wave.

    Returns
    -------
    numpy.ndarray
        Shape ``(l_max + 1,)``, in :math:`(-\pi/2, \pi/2]`.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(hard_sphere_phase_shifts(0.4, 1.0, 0)[0], -0.4))
    True
    """
    ells = np.arange(l_max + 1)
    return np.arctan(spherical_jn(ells, k * a) / spherical_yn(ells, k * a))


def partial_wave_amplitude(theta: np.ndarray, k: float, deltas: np.ndarray) -> np.ndarray:
    r"""Scattering amplitude from phase shifts, :math:`f(\theta) = \frac1k\sum_\ell (2\ell+1)e^{i\delta_\ell}\sin\delta_\ell\,P_\ell(\cos\theta)`.

    (Faxen and Holtsmark, Z. Phys. 45, 307 (1927); Sakurai, Eq. 7.6.18.)

    Parameters
    ----------
    theta : array_like
        Scattering angles (radians).
    k : float
        Wavenumber.
    deltas : array_like
        Phase shifts :math:`\delta_0, \delta_1, \dots`.

    Returns
    -------
    numpy.ndarray
        Complex amplitude.

    Examples
    --------
    Pure s-wave scattering is isotropic:

    >>> import numpy as np
    >>> f = partial_wave_amplitude(np.array([0.0, np.pi]), 1.0, [np.pi / 2])
    >>> bool(np.allclose(f, [1j, 1j]))
    True
    """
    cos_t = np.cos(np.asarray(theta, dtype=float))
    deltas = np.asarray(deltas, dtype=float)
    ells = np.arange(len(deltas))
    coeff = (2 * ells + 1) * np.exp(1j * deltas) * np.sin(deltas) / k
    return sum(c * eval_legendre(ell, cos_t) for ell, c in zip(ells, coeff))


def partial_wave_cross_section(k: float, deltas: np.ndarray) -> float:
    r"""Total cross section :math:`\sigma = \frac{4\pi}{k^2}\sum_\ell (2\ell+1)\sin^2\delta_\ell`.

    Equal, by the optical theorem, to :math:`\frac{4\pi}{k}\,\mathrm{Im}\,f(0)`
    (Sakurai, Eq. 7.6.20).

    Parameters
    ----------
    k : float
        Wavenumber.
    deltas : array_like
        Phase shifts.

    Returns
    -------
    float

    Examples
    --------
    The unitarity limit of a resonant s wave is :math:`4\pi/k^2`:

    >>> import numpy as np
    >>> bool(np.isclose(partial_wave_cross_section(2.0, [np.pi / 2]), np.pi))
    True
    """
    deltas = np.asarray(deltas, dtype=float)
    ells = np.arange(len(deltas))
    return float(4 * np.pi / k**2 * np.sum((2 * ells + 1) * np.sin(deltas) ** 2))
