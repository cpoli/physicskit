r"""Lattice dynamics: phonon dispersion of 1D and 2D lattices and the heat capacity of phonons.

Units :math:`\hbar = k_B = 1`: phonon frequencies :math:`\omega` are also
energies, and temperatures are measured in the same units. Heat
capacities are per atom (or per mode) in units of :math:`k_B`.

- :func:`monatomic_chain_dispersion`, :func:`diatomic_chain_dispersion` --
  the harmonic chain with one or two atoms per cell (Born and von Karman,
  Phys. Z. 13, 297 (1912)).
- :func:`square_lattice_dynamical_matrix`,
  :func:`square_lattice_phonon_dispersion` -- a 2D square lattice with
  nearest- and next-nearest-neighbor central springs.
- :func:`lattice_heat_capacity` -- heat capacity of a set of phonon modes
  (e.g. a sampled dispersion) from Einstein's mode formula.
- :func:`debye_heat_capacity`, :func:`debye_temperature` -- Debye's
  continuum model and its :math:`T^3` law (Debye, Ann. Phys. 344, 789
  (1912)).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

__all__ = [
    "monatomic_chain_dispersion",
    "diatomic_chain_dispersion",
    "square_lattice_dynamical_matrix",
    "square_lattice_phonon_dispersion",
    "lattice_heat_capacity",
    "debye_heat_capacity",
    "debye_temperature",
]


def monatomic_chain_dispersion(k: np.ndarray, K: float = 1.0, m: float = 1.0, a: float = 1.0) -> np.ndarray:
    r"""Phonon dispersion of a monatomic harmonic chain, :math:`\omega(k) = 2\sqrt{K/m}\,|\sin(ka/2)|`.

    Masses :math:`m` joined by springs :math:`K` at spacing :math:`a`
    (Ashcroft and Mermin, *Solid State Physics*, Eq. 22.29). At long
    wavelengths :math:`\omega \approx c|k|` with sound speed
    :math:`c = a\sqrt{K/m}`; at the zone edge :math:`k = \pi/a` the group
    velocity vanishes.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    K : float, default=1.0
        Spring constant.
    m : float, default=1.0
        Atomic mass.
    a : float, default=1.0
        Lattice spacing.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> float(monatomic_chain_dispersion(np.pi, K=4.0))
    4.0
    """
    return 2 * np.sqrt(K / m) * np.abs(np.sin(np.asarray(k, dtype=float) * a / 2))


def diatomic_chain_dispersion(k: np.ndarray, K: float = 1.0, m1: float = 1.0, m2: float = 2.0, a: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    r"""Acoustic and optical branches of a diatomic harmonic chain.

    Alternating masses :math:`m_1, m_2` joined by identical springs
    :math:`K`, with unit-cell length :math:`a`:

    .. math::

        \omega_\pm^2 = K\left(\frac{1}{m_1} + \frac{1}{m_2}\right)
            \pm K\sqrt{\left(\frac{1}{m_1} + \frac{1}{m_2}\right)^2
            - \frac{4\sin^2(ka/2)}{m_1 m_2}}

    (Kittel, *Introduction to Solid State Physics*, 8th ed., Eq. 4.20).
    The acoustic branch has sound speed :math:`c = a\sqrt{K/(2(m_1+m_2))}`;
    the optical branch starts at :math:`\sqrt{2K(1/m_1+1/m_2)}`, and a gap
    opens at the zone boundary between :math:`\sqrt{2K/m_{\max}}` and
    :math:`\sqrt{2K/m_{\min}}`.

    Parameters
    ----------
    k : array_like
        Wavenumber.
    K : float, default=1.0
        Spring constant.
    m1, m2 : float, default=1.0, 2.0
        The two atomic masses.
    a : float, default=1.0
        Unit-cell length.

    Returns
    -------
    acoustic, optical : numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> ac, op = diatomic_chain_dispersion(np.pi, K=1.0, m1=1.0, m2=2.0)
    >>> bool(np.isclose(ac, 1.0) and np.isclose(op, np.sqrt(2.0)))
    True
    """
    k = np.asarray(k, dtype=float)
    s = 1 / m1 + 1 / m2
    gap_term = 4 * np.sin(k * a / 2) ** 2 / (m1 * m2)
    root = np.sqrt(np.maximum(s**2 - gap_term, 0.0))
    # acoustic branch as K (s^2 - root^2)/(s + root): no cancellation at small k
    return np.sqrt(K * gap_term / (s + root)), np.sqrt(K * (s + root))


def square_lattice_dynamical_matrix(kx: np.ndarray, ky: np.ndarray, K1: float = 1.0, K2: float = 0.5, m: float = 1.0, a: float = 1.0) -> np.ndarray:
    r"""Dynamical matrix of a square lattice with nearest (:math:`K_1`) and next-nearest (:math:`K_2`) central springs.

    For central-force springs to neighbors :math:`\mathbf R` with unit
    vectors :math:`\hat{\mathbf e}`,
    :math:`D(\mathbf k) = \sum_{\mathbf R} \frac{K_R}{m}(1 - \cos\mathbf k\cdot\mathbf R)\,
    \hat{\mathbf e}\hat{\mathbf e}^T` (Ashcroft and Mermin, Eq. 22.58),
    which here gives

    .. math::

        D = \frac{2K_1}{m}\begin{pmatrix} 1 - \cos k_xa & 0 \\ 0 & 1 - \cos k_ya \end{pmatrix}
          + \frac{K_2}{m}\left[(1 - \cos(k_x + k_y)a)\begin{pmatrix}1&1\\1&1\end{pmatrix}
          + (1 - \cos(k_x - k_y)a)\begin{pmatrix}1&-1\\-1&1\end{pmatrix}\right].

    The diagonal springs give the lattice its shear stiffness: with
    :math:`K_2 = 0` the transverse branch is dispersionless along the axes.

    Parameters
    ----------
    kx, ky : array_like
        Wavevector components (broadcast together).
    K1, K2 : float, default=1.0, 0.5
        Nearest- and next-nearest-neighbor spring constants.
    m : float, default=1.0
        Atomic mass.
    a : float, default=1.0
        Lattice constant.

    Returns
    -------
    numpy.ndarray
        Shape ``broadcast(kx, ky).shape + (2, 2)``, real symmetric.

    Examples
    --------
    >>> D = square_lattice_dynamical_matrix(0.0, 0.0)
    >>> D.tolist()
    [[0.0, 0.0], [0.0, 0.0]]
    """
    kx, ky = np.broadcast_arrays(np.asarray(kx, dtype=float), np.asarray(ky, dtype=float))

    def one_minus_cos(x):  # 2 sin^2(x/2), accurate at small x
        return 2 * np.sin(x / 2) ** 2

    D = np.zeros(kx.shape + (2, 2))
    D[..., 0, 0] = 2 * K1 / m * one_minus_cos(kx * a)
    D[..., 1, 1] = 2 * K1 / m * one_minus_cos(ky * a)
    plus = K2 / m * one_minus_cos((kx + ky) * a)
    minus = K2 / m * one_minus_cos((kx - ky) * a)
    D[..., 0, 0] += plus + minus
    D[..., 1, 1] += plus + minus
    D[..., 0, 1] += plus - minus
    D[..., 1, 0] += plus - minus
    return D


def square_lattice_phonon_dispersion(kx: np.ndarray, ky: np.ndarray, K1: float = 1.0, K2: float = 0.5, m: float = 1.0, a: float = 1.0) -> np.ndarray:
    r"""Phonon frequencies :math:`\omega_s(\mathbf k) = \sqrt{\lambda_s(D(\mathbf k))}` of the square lattice.

    Along :math:`\mathbf k \parallel \hat x` the long-wavelength sound
    speeds are :math:`c_L = a\sqrt{(K_1 + K_2)/m}` (longitudinal) and
    :math:`c_T = a\sqrt{K_2/m}` (transverse).

    Parameters
    ----------
    kx, ky : array_like
        Wavevector components.
    K1, K2 : float, default=1.0, 0.5
        Spring constants.
    m : float, default=1.0
        Atomic mass.
    a : float, default=1.0
        Lattice constant.

    Returns
    -------
    numpy.ndarray
        Shape ``broadcast(kx, ky).shape + (2,)``, ascending per wavevector.

    Examples
    --------
    At :math:`(\pi, \pi)/a` both branches have :math:`\omega^2 = 4K_1/m`:

    >>> import numpy as np
    >>> bool(np.allclose(square_lattice_phonon_dispersion(np.pi, np.pi), 2.0))
    True
    """
    D = square_lattice_dynamical_matrix(kx, ky, K1, K2, m, a)
    return np.sqrt(np.maximum(np.linalg.eigvalsh(D), 0.0))


def lattice_heat_capacity(frequencies: np.ndarray, T: np.ndarray) -> np.ndarray:
    r"""Average heat capacity per phonon mode, :math:`C/(N_\mathrm{modes} k_B)`.

    Each harmonic mode of frequency :math:`\omega` contributes Einstein's

    .. math::

        c(\omega, T) = \left(\frac{\omega}{T}\right)^2
            \frac{e^{\omega/T}}{(e^{\omega/T} - 1)^2}

    (Einstein, Ann. Phys. 327, 180 (1907)), tending to 1 (equipartition)
    at high :math:`T`. Summing over a Brillouin-zone sample of a phonon
    dispersion gives the lattice heat capacity without Debye's continuum
    approximation.

    Parameters
    ----------
    frequencies : array_like
        Mode frequencies (any shape; zero-frequency modes contribute 1).
    T : array_like
        Temperatures.

    Returns
    -------
    numpy.ndarray
        Shape of ``T``.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(lattice_heat_capacity(np.array([1.0]), 1000.0)), 6)
    1.0
    """
    w = np.asarray(frequencies, dtype=float).ravel()
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))
    x = w[None, :] / T_arr[:, None]
    with np.errstate(over="ignore", invalid="ignore"):
        # x^2 e^x/(e^x-1)^2 = (x / (2 sinh(x/2)))^2, with limit 1 as x -> 0
        c = np.where(x < 1e-8, 1.0, (x / (2 * np.sinh(x / 2))) ** 2)
    c = np.nan_to_num(c, nan=0.0)
    out = c.mean(axis=1)
    return out if np.ndim(T) else out[0]


def debye_heat_capacity(T: np.ndarray, theta_D: float) -> np.ndarray:
    r"""Debye heat capacity per atom of a 3D solid, in units of :math:`k_B`.

    .. math::

        \frac{C}{Nk_B} = 9\left(\frac{T}{\Theta_D}\right)^3
            \int_0^{\Theta_D/T}\frac{x^4 e^x}{(e^x - 1)^2}\,dx

    (Debye 1912; Kittel, Eq. 5.35), interpolating between the
    Dulong-Petit value 3 for :math:`T \gg \Theta_D` and the Debye
    :math:`T^3` law :math:`\frac{12\pi^4}{5}(T/\Theta_D)^3` for
    :math:`T \ll \Theta_D`.

    Parameters
    ----------
    T : array_like
        Temperature (same units as :math:`\Theta_D`).
    theta_D : float
        Debye temperature.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> T = 0.01
    >>> bool(np.isclose(debye_heat_capacity(T, 1.0), 12 * np.pi**4 / 5 * T**3, rtol=1e-6))
    True
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=float))

    def integrand(x):
        return x**4 / (4 * np.sinh(x / 2) ** 2) if x > 0 else 0.0  # x^4 e^x/(e^x-1)^2

    out = np.array([9 * (t / theta_D) ** 3 * quad(integrand, 0, min(theta_D / t, 700.0), limit=200)[0] if t > 0 else 0.0 for t in T_arr])
    return out if np.ndim(T) else out[0]


def debye_temperature(sound_speed: float, number_density: float) -> float:
    r"""Debye temperature :math:`\Theta_D = c\,(6\pi^2 n)^{1/3}` (units :math:`\hbar = k_B = 1`).

    Cutting the linear dispersion :math:`\omega = ck` off at the Debye
    wavenumber :math:`k_D = (6\pi^2 n)^{1/3}`, which holds exactly
    :math:`N` modes per polarization (Kittel, Eq. 5.30), with :math:`c` the
    (suitably averaged) sound speed.

    Parameters
    ----------
    sound_speed : float
        Sound speed :math:`c`.
    number_density : float
        Atoms per unit volume :math:`n`.

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(debye_temperature(1.0, 1 / (6 * np.pi**2)), 1.0))
    True
    """
    return float(sound_speed * (6 * np.pi**2 * number_density) ** (1 / 3))
