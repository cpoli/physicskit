r"""Electronic transport: the Drude model and the Boltzmann equation in the relaxation-time approximation.

Units :math:`k_B = 1`. Charges, masses, and densities are left general
(:math:`q = -1` for electrons in units of :math:`e`).

- :func:`drude_conductivity`, :func:`drude_ac_conductivity`,
  :func:`drude_conductivity_tensor`, :func:`hall_coefficient` -- Drude's
  classical free-electron gas (Drude, Ann. Phys. 306, 566 (1900)).
- :func:`fermi_window`, :func:`boltzmann_transport` -- the linearized
  Boltzmann equation with a constant relaxation time for any band
  structure sampled on a :math:`k`-grid, giving the conductivity, Seebeck
  coefficient, and electronic thermal conductivity (Sommerfeld, Z. Phys.
  47, 1 (1928); Ashcroft and Mermin, *Solid State Physics*, Ch. 13).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

__all__ = [
    "drude_conductivity",
    "drude_ac_conductivity",
    "drude_conductivity_tensor",
    "hall_coefficient",
    "fermi_window",
    "BoltzmannTransport",
    "boltzmann_transport",
]


def drude_conductivity(n: float, tau: float, q: float = -1.0, m: float = 1.0) -> float:
    r"""Drude DC conductivity :math:`\sigma_0 = nq^2\tau/m`.

    Carriers of charge :math:`q` and mass :math:`m`, accelerated by the
    field and randomized by collisions every :math:`\tau` on average
    (Drude 1900; Ashcroft and Mermin, Eq. 1.6).

    Parameters
    ----------
    n : float
        Carrier density.
    tau : float
        Relaxation (mean free) time.
    q : float, default=-1.0
        Carrier charge.
    m : float, default=1.0
        Carrier mass.

    Returns
    -------
    float

    Examples
    --------
    >>> drude_conductivity(n=2.0, tau=3.0)
    6.0
    """
    return n * q**2 * tau / m


def drude_ac_conductivity(omega: np.ndarray, n: float, tau: float, q: float = -1.0, m: float = 1.0) -> np.ndarray:
    r"""Drude AC conductivity :math:`\sigma(\omega) = \sigma_0/(1 - i\omega\tau)`.

    For fields :math:`\propto e^{-i\omega t}` (Ashcroft and Mermin, Eq.
    1.29). The real part is a Lorentzian of width :math:`1/\tau`, whose
    integral obeys the f-sum rule
    :math:`\int_0^\infty \mathrm{Re}\,\sigma\,d\omega = \pi nq^2/(2m)`.

    Parameters
    ----------
    omega : array_like
        Angular frequency.
    n, tau, q, m : float
        As in :func:`drude_conductivity`.

    Returns
    -------
    numpy.ndarray
        Complex conductivity.

    Examples
    --------
    >>> complex(drude_ac_conductivity(1.0, n=1.0, tau=1.0))
    (0.5+0.5j)
    """
    return drude_conductivity(n, tau, q, m) / (1 - 1j * np.asarray(omega, dtype=float) * tau)


def drude_conductivity_tensor(B: float, n: float, tau: float, q: float = -1.0, m: float = 1.0) -> np.ndarray:
    r"""In-plane Drude conductivity tensor in a perpendicular magnetic field :math:`B\hat z`.

    Solving :math:`m\dot{\mathbf v} = q(\mathbf E + \mathbf v\times\mathbf B)
    - m\mathbf v/\tau` in steady state gives

    .. math::

        \sigma = \frac{\sigma_0}{1 + (\omega_c\tau)^2}
            \begin{pmatrix} 1 & \omega_c\tau \\ -\omega_c\tau & 1 \end{pmatrix},
        \qquad \omega_c = \frac{qB}{m},

    whose inverse, the resistivity, has :math:`\rho_{xx} = 1/\sigma_0`
    independent of :math:`B` (no magnetoresistance) and
    :math:`\rho_{yx} = B/(nq)` (Ashcroft and Mermin, Eqs. 1.40-1.44).

    Parameters
    ----------
    B : float
        Magnetic field along :math:`z`.
    n, tau, q, m : float
        As in :func:`drude_conductivity`.

    Returns
    -------
    numpy.ndarray
        Shape ``(2, 2)``.

    Examples
    --------
    >>> import numpy as np
    >>> rho = np.linalg.inv(drude_conductivity_tensor(B=2.0, n=0.5, tau=1.0))
    >>> bool(np.isclose(rho[1, 0], 2.0 / (0.5 * -1.0)))
    True
    """
    beta = q * B * tau / m
    return drude_conductivity(n, tau, q, m) / (1 + beta**2) * np.array([[1.0, beta], [-beta, 1.0]])


def hall_coefficient(n: float, q: float = -1.0) -> float:
    r"""Drude Hall coefficient :math:`R_H = E_y/(j_x B) = 1/(nq)`.

    Negative for electrons; its measured sign revealed hole conduction in
    metals such as Be and Al that the free-electron model could not
    explain (Ashcroft and Mermin, Table 1.4).

    Parameters
    ----------
    n : float
        Carrier density.
    q : float, default=-1.0
        Carrier charge.

    Returns
    -------
    float

    Examples
    --------
    >>> hall_coefficient(0.25)
    -4.0
    """
    return 1.0 / (n * q)


def fermi_window(energy: np.ndarray, mu: float, T: float) -> np.ndarray:
    r"""The Fermi window :math:`-\partial f/\partial\varepsilon = \dfrac{1}{4T\cosh^2[(\varepsilon - \mu)/2T]}`.

    Parameters
    ----------
    energy : array_like
        Energies.
    mu : float
        Chemical potential.
    T : float
        Temperature, :math:`T > 0`.

    Returns
    -------
    numpy.ndarray
        Integrates to 1 over energy.

    Examples
    --------
    >>> float(fermi_window(0.0, mu=0.0, T=0.25))
    1.0
    """
    x = np.clip((np.asarray(energy, dtype=float) - mu) / (2 * T), -350, 350)
    return 1.0 / (4 * T * np.cosh(x) ** 2)


@dataclass
class BoltzmannTransport:
    r"""Transport coefficients returned by :func:`boltzmann_transport`.

    Attributes
    ----------
    sigma : numpy.ndarray
        Electrical conductivity tensor, shape ``(d, d)``.
    seebeck : numpy.ndarray
        Seebeck (thermopower) tensor :math:`S`, with :math:`\mathbf E =
        S\nabla T` at zero current.
    kappa : numpy.ndarray
        Electronic thermal conductivity tensor at zero electric current.
    density : float
        Carrier density :math:`\int f\,d^dk\,g_s/(2\pi)^d`.
    T : float
        Temperature.
    """

    sigma: np.ndarray
    seebeck: np.ndarray
    kappa: np.ndarray
    density: float
    T: float

    @property
    def lorenz_number(self) -> float:
        r"""Lorenz ratio :math:`L = \kappa_{xx}/(\sigma_{xx}T)`; :math:`\pi^2/(3q^2)` for a degenerate metal."""
        return float(self.kappa[0, 0] / (self.sigma[0, 0] * self.T))


def boltzmann_transport(
    energies: np.ndarray,
    velocities: np.ndarray,
    cell_volume: float,
    mu: float,
    T: float,
    tau: float = 1.0,
    q: float = -1.0,
    spin_degeneracy: float = 2.0,
) -> BoltzmannTransport:
    r"""Boltzmann transport coefficients in the constant relaxation-time approximation.

    For a band :math:`\varepsilon_{\mathbf k}` with velocities
    :math:`\mathbf v_{\mathbf k} = \nabla_{\mathbf k}\varepsilon`, define the
    moments

    .. math::

        \mathcal K_n = \tau \int \frac{g_s\,d^dk}{(2\pi)^d}
            \left(-\frac{\partial f}{\partial\varepsilon}\right)
            \mathbf v_{\mathbf k}\mathbf v_{\mathbf k}^T
            (\varepsilon_{\mathbf k} - \mu)^n.

    Then (Ashcroft and Mermin, Eqs. 13.37-13.47, for general charge
    :math:`q`)

    .. math::

        \sigma = q^2\mathcal K_0, \qquad
        S = \frac{1}{qT}\,\mathcal K_0^{-1}\mathcal K_1, \qquad
        \kappa = \frac{1}{T}\left(\mathcal K_2 - \mathcal K_1\mathcal K_0^{-1}\mathcal K_1\right).

    For a degenerate metal the Sommerfeld expansion gives the
    Wiedemann-Franz law :math:`\kappa/(\sigma T) = \pi^2/(3q^2)` and Mott's
    formula :math:`S = \frac{\pi^2 T}{3q}\,\frac{d\ln\sigma(\varepsilon)}{d\varepsilon}\big|_\mu`;
    for a parabolic band :math:`\sigma = nq^2\tau/m`, Drude's result with
    :math:`n` the actual carrier density.

    Parameters
    ----------
    energies : numpy.ndarray
        Band energies on a uniform :math:`k`-grid, shape ``(M,)`` (flatten
        multi-band or multi-dimensional grids).
    velocities : numpy.ndarray
        Group velocities at the same points, shape ``(M, d)``.
    cell_volume : float
        :math:`k`-space volume per grid point, :math:`\Delta^d k`.
    mu : float
        Chemical potential.
    T : float
        Temperature, :math:`T > 0`.
    tau : float, default=1.0
        Relaxation time.
    q : float, default=-1.0
        Carrier charge.
    spin_degeneracy : float, default=2.0
        :math:`g_s`.

    Returns
    -------
    BoltzmannTransport

    Examples
    --------
    A 1D parabolic band at low temperature reproduces Drude's
    :math:`\sigma = nq^2\tau/m`:

    >>> import numpy as np
    >>> k = np.linspace(-4, 4, 40001)
    >>> res = boltzmann_transport(k**2 / 2, k[:, None], k[1] - k[0], mu=1.0, T=0.01)
    >>> bool(np.isclose(res.sigma[0, 0], res.density, rtol=1e-3))
    True
    """
    eps = np.asarray(energies, dtype=float).ravel()
    v = np.asarray(velocities, dtype=float).reshape(eps.size, -1)
    d = v.shape[1]
    w = spin_degeneracy * cell_volume / (2 * np.pi) ** d
    window = fermi_window(eps, mu, T) * w
    de = eps - mu
    K = [tau * np.einsum("m,mi,mj->ij", window * de**n, v, v) for n in range(3)]
    K0_inv = np.linalg.inv(K[0])
    with np.errstate(over="ignore"):
        occupation = 1.0 / (np.exp(np.clip(de / T, -700, 700)) + 1.0)
    return BoltzmannTransport(
        sigma=q**2 * K[0],
        seebeck=K0_inv @ K[1] / (q * T),
        kappa=(K[2] - K[1] @ K0_inv @ K[1]) / T,
        density=float(w * occupation.sum()),
        T=T,
    )
