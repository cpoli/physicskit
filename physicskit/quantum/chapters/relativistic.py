r"""Relativistic quantum mechanics: the Klein-Gordon and Dirac equations.

Units :math:`\hbar = c = 1`, so energies and momenta are measured in the
same units as the mass :math:`m` (energies of the Dirac hydrogen atom are
returned in units of :math:`mc^2`).

- :func:`klein_gordon_dispersion`, :func:`klein_gordon_plane_wave` -- free
  solutions of :math:`(\partial_t^2 - \nabla^2 + m^2)\phi = 0`.
- :func:`gamma_matrices`, :func:`dirac_hamiltonian`,
  :func:`dirac_plane_wave_spinor` -- the Dirac equation in the Dirac
  representation and its free plane-wave spinors.
- :func:`dirac_hydrogen_energy`, :func:`fine_structure_expansion` -- the
  exact Dirac-Coulomb levels and their :math:`(Z\alpha)^4` fine-structure
  expansion.
- :func:`dirac_step_scattering`, :func:`klein_gordon_step_scattering` --
  reflection and transmission at a potential step, including the Klein
  paradox regime :math:`V_0 > E + m`.
"""

from __future__ import annotations

import numpy as np
from scipy.constants import fine_structure

__all__ = [
    "klein_gordon_dispersion",
    "klein_gordon_plane_wave",
    "gamma_matrices",
    "dirac_hamiltonian",
    "dirac_plane_wave_spinor",
    "dirac_hydrogen_energy",
    "fine_structure_expansion",
    "dirac_step_scattering",
    "klein_gordon_step_scattering",
]

_PAULI = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)
_I2 = np.eye(2, dtype=complex)
_Z2 = np.zeros((2, 2), dtype=complex)


def klein_gordon_dispersion(p: np.ndarray | float, m: float = 1.0) -> np.ndarray:
    r"""Positive-energy branch :math:`E = \sqrt{p^2 + m^2}` of the Klein-Gordon equation.

    Substituting :math:`\phi = e^{i(\mathbf p\cdot\mathbf x - Et)}` into
    :math:`(\partial_t^2 - \nabla^2 + m^2)\phi = 0` (Klein, Z. Phys. 37, 895
    (1926); Gordon, Z. Phys. 40, 117 (1926)) gives :math:`E^2 = p^2 + m^2`,
    with solutions of both signs :math:`E = \pm\sqrt{p^2+m^2}`.

    Parameters
    ----------
    p : array_like
        Momentum magnitude :math:`|\mathbf p|`.
    m : float, default=1.0
        Mass.

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> float(klein_gordon_dispersion(0.75, m=1.0))
    1.25
    """
    p = np.asarray(p, dtype=float)
    return np.sqrt(p**2 + m**2)


def klein_gordon_plane_wave(x: np.ndarray, t: float, p: float, m: float = 1.0, energy_sign: int = 1) -> np.ndarray:
    r"""Free 1D Klein-Gordon plane wave :math:`\phi(x,t) = e^{i(px - Et)}`, :math:`E = \pm\sqrt{p^2+m^2}`.

    Parameters
    ----------
    x : numpy.ndarray
        Positions.
    t : float
        Time.
    p : float
        Momentum.
    m : float, default=1.0
        Mass.
    energy_sign : {+1, -1}, default=+1
        Sign of the energy.

    Returns
    -------
    numpy.ndarray
        Complex amplitudes.

    Examples
    --------
    >>> import numpy as np
    >>> phi = klein_gordon_plane_wave(np.array([0.0]), t=0.0, p=1.0)
    >>> phi
    array([1.+0.j])
    """
    E = energy_sign * klein_gordon_dispersion(p, m)
    return np.exp(1j * (p * np.asarray(x) - E * t))


def gamma_matrices() -> np.ndarray:
    r"""Dirac gamma matrices :math:`\gamma^\mu` in the Dirac (standard) representation.

    .. math::

        \gamma^0 = \begin{pmatrix} I & 0 \\ 0 & -I \end{pmatrix},\qquad
        \gamma^i = \begin{pmatrix} 0 & \sigma_i \\ -\sigma_i & 0 \end{pmatrix},

    satisfying :math:`\{\gamma^\mu, \gamma^\nu\} = 2 g^{\mu\nu}` with
    :math:`g = \mathrm{diag}(1,-1,-1,-1)` (Bjorken and Drell, *Relativistic
    Quantum Mechanics*, 1964, Eq. 2.7).

    Returns
    -------
    numpy.ndarray
        Shape ``(4, 4, 4)``: ``gamma[mu]`` is :math:`\gamma^\mu`.

    Examples
    --------
    >>> import numpy as np
    >>> g = gamma_matrices()
    >>> bool(np.allclose(g[1] @ g[1], -np.eye(4)))
    True
    """
    g0 = np.block([[_I2, _Z2], [_Z2, -_I2]])
    gs = [np.block([[_Z2, s], [-s, _Z2]]) for s in _PAULI]
    return np.array([g0, *gs])


def dirac_hamiltonian(p: np.ndarray, m: float = 1.0) -> np.ndarray:
    r"""Free Dirac Hamiltonian :math:`H = \boldsymbol\alpha\cdot\mathbf p + \beta m` for momentum :math:`\mathbf p`.

    With :math:`\beta = \gamma^0` and :math:`\alpha_i = \gamma^0\gamma^i`
    (Dirac, Proc. R. Soc. A 117, 610 (1928)). Its eigenvalues are
    :math:`\pm\sqrt{p^2 + m^2}`, each doubly degenerate (spin).

    Parameters
    ----------
    p : array_like
        Momentum 3-vector.
    m : float, default=1.0
        Mass.

    Returns
    -------
    numpy.ndarray
        Shape ``(4, 4)``, Hermitian.

    Examples
    --------
    >>> import numpy as np
    >>> np.linalg.eigvalsh(dirac_hamiltonian([0.0, 0.0, 0.75])).round(12)
    array([-1.25, -1.25,  1.25,  1.25])
    """
    g = gamma_matrices()
    p = np.asarray(p, dtype=float)
    return sum(p[i] * (g[0] @ g[i + 1]) for i in range(3)) + m * g[0]


def dirac_plane_wave_spinor(p: np.ndarray, m: float = 1.0, spin: int = 1, energy_sign: int = 1) -> np.ndarray:
    r"""Free Dirac plane-wave spinor, eigenvector of :func:`dirac_hamiltonian` with energy :math:`\pm E`.

    .. math::

        u_s(\mathbf p) = \sqrt{E + m}\begin{pmatrix} \chi_s \\
            \dfrac{\boldsymbol\sigma\cdot\mathbf p}{E + m}\chi_s \end{pmatrix},
        \qquad
        w_s(\mathbf p) = \sqrt{E + m}\begin{pmatrix}
            -\dfrac{\boldsymbol\sigma\cdot\mathbf p}{E + m}\chi_s \\ \chi_s
            \end{pmatrix},

    for energies :math:`+E` and :math:`-E` respectively, :math:`E =
    \sqrt{p^2+m^2}`, normalized to :math:`\psi^\dagger\psi = 2E`
    (Bjorken and Drell, Eq. 3.7; for positive energy
    :math:`\bar u u = 2m`).

    Parameters
    ----------
    p : array_like
        Momentum 3-vector.
    m : float, default=1.0
        Mass.
    spin : {+1, -1}, default=+1
        Two-component spinor :math:`\chi_{+} = (1, 0)^T` or
        :math:`\chi_{-} = (0, 1)^T`.
    energy_sign : {+1, -1}, default=+1
        Positive- or negative-energy solution.

    Returns
    -------
    numpy.ndarray
        Shape ``(4,)``, complex.

    Examples
    --------
    >>> import numpy as np
    >>> p = np.array([0.3, -0.2, 0.5])
    >>> u = dirac_plane_wave_spinor(p)
    >>> E = np.sqrt(p @ p + 1)
    >>> bool(np.allclose(dirac_hamiltonian(p) @ u, E * u))
    True
    """
    p = np.asarray(p, dtype=float)
    E = float(np.sqrt(p @ p + m**2))
    chi = np.array([1, 0], dtype=complex) if spin == 1 else np.array([0, 1], dtype=complex)
    sp = np.einsum("i,ijk->jk", p, _PAULI)
    small = sp @ chi / (E + m)
    parts = (chi, small) if energy_sign == 1 else (-small, chi)
    return np.sqrt(E + m) * np.concatenate(parts)


def dirac_hydrogen_energy(n: int, j: float, Z: float = 1.0, alpha: float = fine_structure) -> float:
    r"""Exact Dirac-Coulomb energy level, in units of :math:`mc^2` (rest energy included).

    .. math::

        \frac{E_{nj}}{mc^2} = \left[1 + \left(\frac{Z\alpha}
            {n - (j + \tfrac12) + \sqrt{(j + \tfrac12)^2 - (Z\alpha)^2}}
            \right)^2\right]^{-1/2}

    (Darwin, Proc. R. Soc. A 118, 654 (1928); Gordon, Z. Phys. 48, 11
    (1928); Bjorken and Drell, Eq. 4.29). It depends on :math:`n` and
    :math:`j` only, so :math:`2S_{1/2}` and :math:`2P_{1/2}` stay degenerate
    (the Lamb shift lifts this). The ground state is
    :math:`E_{1,1/2} = mc^2\sqrt{1 - (Z\alpha)^2}`.

    Parameters
    ----------
    n : int
        Principal quantum number, :math:`n \ge 1`.
    j : float
        Total angular momentum, :math:`\tfrac12 \le j \le n - \tfrac12`.
    Z : float, default=1.0
        Nuclear charge.
    alpha : float, default=scipy.constants.fine_structure
        Fine-structure constant.

    Returns
    -------
    float
        :math:`E/(mc^2)`.

    Raises
    ------
    ValueError
        If :math:`j` is out of range or :math:`Z\alpha \ge j + 1/2`.

    Examples
    --------
    >>> import numpy as np
    >>> bool(np.isclose(dirac_hydrogen_energy(1, 0.5, Z=50), np.sqrt(1 - (50 * 7.2973525693e-3) ** 2)))
    True
    """
    kappa = j + 0.5
    if not (0.5 <= j <= n - 0.5) or abs(kappa - round(kappa)) > 1e-12:
        raise ValueError("need half-integer j with 1/2 <= j <= n - 1/2")
    za = Z * alpha
    if za >= kappa:
        raise ValueError("Z*alpha must be below j + 1/2")
    return float(1.0 / np.sqrt(1.0 + (za / (n - kappa + np.sqrt(kappa**2 - za**2))) ** 2))


def fine_structure_expansion(n: int, j: float, Z: float = 1.0, alpha: float = fine_structure) -> float:
    r"""Binding energy through order :math:`(Z\alpha)^4`, in units of :math:`mc^2`.

    .. math::

        \frac{E_{nj} - mc^2}{mc^2} \approx -\frac{(Z\alpha)^2}{2n^2}
            - \frac{(Z\alpha)^4}{2n^4}\left(\frac{n}{j + 1/2} - \frac34\right)

    -- the Bohr level plus the fine-structure correction (relativistic
    kinetic energy, spin-orbit, and Darwin terms combined; Sommerfeld 1916,
    Bethe and Salpeter, *Quantum Mechanics of One- and Two-Electron Atoms*,
    1957, Eq. 17.13).

    Parameters
    ----------
    n : int
        Principal quantum number.
    j : float
        Total angular momentum.
    Z : float, default=1.0
        Nuclear charge.
    alpha : float, default=scipy.constants.fine_structure
        Fine-structure constant.

    Returns
    -------
    float

    Examples
    --------
    The :math:`2P_{3/2}`-:math:`2P_{1/2}` splitting is :math:`\alpha^4/32`:

    >>> import numpy as np
    >>> a = 7.2973525693e-3
    >>> split = fine_structure_expansion(2, 1.5, alpha=a) - fine_structure_expansion(2, 0.5, alpha=a)
    >>> bool(np.isclose(split, a**4 / 32))
    True
    """
    za = Z * alpha
    return float(-(za**2) / (2 * n**2) - za**4 / (2 * n**4) * (n / (j + 0.5) - 0.75))


def _step_momenta(E: np.ndarray, V0: np.ndarray, m: float, group_velocity: bool) -> tuple[np.ndarray, np.ndarray]:
    p = np.sqrt(E**2 - m**2 + 0j)
    q = np.sqrt((E - V0) ** 2 - m**2 + 0j)
    # propagating: pick the transmitted wave moving *away* from the step;
    # evanescent: pick the decaying branch (Im q > 0)
    propagating = np.abs(q.imag) < 1e-300
    if group_velocity:
        q = np.where(propagating, np.sign(E - V0) * np.abs(q.real), q)
    return p, q


def dirac_step_scattering(E: np.ndarray, V0: np.ndarray, m: float = 1.0, group_velocity: bool = True) -> tuple:
    r"""Reflection and transmission probabilities of a Dirac particle at a potential step.

    A spin-up plane wave of energy :math:`E > m` hits :math:`V(z) = V_0
    \Theta(z)`. Matching the Dirac spinor at :math:`z = 0` gives

    .. math::

        \kappa = \frac{q}{p}\,\frac{E + m}{E - V_0 + m}, \qquad
        R = \left|\frac{1 - \kappa}{1 + \kappa}\right|^2, \qquad
        T = \frac{4\,\mathrm{Re}\,\kappa}{|1 + \kappa|^2},

    with :math:`p = \sqrt{E^2 - m^2}` and :math:`q = \sqrt{(E - V_0)^2 -
    m^2}`, so that :math:`R + T = 1` (Bjorken and Drell, Sec. 3.3; Greiner,
    *Relativistic Quantum Mechanics*, Sec. 13.1). For
    :math:`|E - V_0| < m`, :math:`q` is imaginary and :math:`R = 1`. In the
    **Klein zone** :math:`V_0 > E + m` the transmitted wave is a
    negative-energy (antiparticle) state: choosing :math:`q` so that its
    *group velocity* :math:`q/(E - V_0)` points away from the step gives
    :math:`\kappa > 0` and :math:`0 < R < 1` -- a finite transmission even
    as :math:`V_0 \to \infty`, Klein's paradox (Klein, Z. Phys. 53, 157
    (1929); Dombey and Calogeracos, Phys. Rep. 315, 41 (1999)).

    Parameters
    ----------
    E : array_like
        Energy (including rest mass), :math:`E > m`.
    V0 : array_like
        Step height.
    m : float, default=1.0
        Mass.
    group_velocity : bool, default=True
        If False, use the naive :math:`q > 0` (phase-velocity) choice,
        which in the Klein zone gives :math:`R > 1`, :math:`T < 0`.

    Returns
    -------
    R, T : numpy.ndarray
        Reflection and transmission probabilities (particle-current ratios).

    Examples
    --------
    Far inside the Klein zone :math:`T \to 4\kappa_\infty/(1+\kappa_\infty)^2`
    with :math:`\kappa_\infty = \sqrt{(E+m)/(E-m)}`:

    >>> import numpy as np
    >>> R, T = dirac_step_scattering(2.0, 1e8)
    >>> kinf = np.sqrt(3.0)
    >>> bool(np.isclose(T, 4 * kinf / (1 + kinf) ** 2))
    True
    """
    E = np.asarray(E, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    p, q = _step_momenta(E, V0, m, group_velocity)
    edge = np.abs(q) < 1e-14  # |E - V0| = m: zero transmitted momentum, R = 1
    with np.errstate(divide="ignore", invalid="ignore"):
        kappa = np.where(edge, 0.0, q / p * (E + m) / (E - V0 + m))
    R = np.abs((1 - kappa) / (1 + kappa)) ** 2
    T = 4 * kappa.real / np.abs(1 + kappa) ** 2
    return R, T


def klein_gordon_step_scattering(E: np.ndarray, V0: np.ndarray, m: float = 1.0, group_velocity: bool = True) -> tuple:
    r"""Reflection and transmission of a Klein-Gordon (spin-0) particle at a potential step.

    Continuity of :math:`\phi` and :math:`\phi'` at :math:`z = 0` gives

    .. math::

        R = \left|\frac{p - q}{p + q}\right|^2, \qquad
        T = \frac{4p\,\mathrm{Re}\,q}{|p + q|^2},

    with the transmitted wave chosen to move away from the step (group
    velocity :math:`q/(E - V_0) > 0`). In the Klein zone :math:`V_0 > E + m`
    this makes :math:`q < 0`: the transmitted charge current is negative
    and :math:`R > 1` -- the bosonic "superradiant" Klein paradox, the
    counterpart of the fermionic :math:`R < 1` of
    :func:`dirac_step_scattering` (Greiner, Sec. 1.13; Dombey and
    Calogeracos 1999). At :math:`V_0 = 2E` the transmitted and incident
    momenta cancel, :math:`q = -p`, and :math:`R` and :math:`T` diverge.

    Parameters
    ----------
    E : array_like
        Energy (including rest mass), :math:`E > m`.
    V0 : array_like
        Step height.
    m : float, default=1.0
        Mass.
    group_velocity : bool, default=True
        If False, use the naive :math:`q > 0` choice.

    Returns
    -------
    R, T : numpy.ndarray
        Reflection and transmission coefficients (charge-current ratios),
        :math:`R + T = 1`.

    Examples
    --------
    >>> R, T = klein_gordon_step_scattering(2.0, 5.0)
    >>> bool(R > 1 and T < 0 and abs(R + T - 1) < 1e-12)
    True
    """
    E = np.asarray(E, dtype=float)
    V0 = np.asarray(V0, dtype=float)
    p, q = _step_momenta(E, V0, m, group_velocity)
    R = np.abs((p - q) / (p + q)) ** 2
    T = 4 * p.real * q.real / np.abs(p + q) ** 2
    return R, T
