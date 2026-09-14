r"""Quantum optics in the truncated Fock basis: states, Wigner functions, and Jaynes-Cummings dynamics.

Light is quantized in terms of the harmonic-oscillator (photon-number, or
Fock) states :math:`\lvert n\rangle` of a single cavity mode. This module
works throughout in a Fock basis truncated to a finite dimension
``cutoff`` -- large enough that the state's amplitude on the highest
retained level is negligible -- and provides:

- constructors for the number, coherent, and squeezed states (built from
  the ladder operators of :mod:`physicskit.quantum.core.operators`);
- the Wigner quasi-probability distribution, computed from its closed-form
  matrix elements in the Fock basis (Cahill & Glauber, *Phys. Rev.* **177**,
  1857, 1969), using the dimensionless quadratures

  .. math::

      \hat x = \frac{\hat a + \hat a^\dagger}{\sqrt2}, \qquad
      \hat p = \frac{\hat a - \hat a^\dagger}{i\sqrt2};

- the Jaynes-Cummings model (Jaynes & Cummings, *Proc. IEEE* **51**, 89,
  1963), the fully quantum treatment of a two-level atom coupled to a
  single cavity mode in the rotating-wave approximation, famous for
  predicting vacuum Rabi oscillations and collapse-and-revival -- effects
  with no semiclassical analogue.
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import expm
from scipy.special import eval_genlaguerre, gammaln

from physicskit.optics._compat import trapz
from physicskit.quantum.core.operators import annihilation_operator, creation_operator

__all__ = [
    "fock_state",
    "coherent_state",
    "squeezed_state",
    "compute_wigner_function",
    "wigner_negativity",
    "JaynesCummingsModel",
]


def fock_state(n, cutoff):
    r"""The number (Fock) state :math:`\lvert n\rangle`.

    Parameters
    ----------
    n : int
        Photon number.
    cutoff : int
        Dimension of the truncated Fock space.

    Returns
    -------
    ndarray of shape (cutoff,)
        Complex state vector, all zeros except a ``1`` at index ``n``.

    Raises
    ------
    ValueError
        If ``n`` is negative or ``n >= cutoff``.

    Examples
    --------
    >>> fock_state(1, 4)
    array([0.+0.j, 1.+0.j, 0.+0.j, 0.+0.j])
    """
    if n < 0 or n >= cutoff:
        raise ValueError(f"n={n} must satisfy 0 <= n < cutoff={cutoff}.")
    state = np.zeros(cutoff, dtype=complex)
    state[n] = 1.0
    return state


def coherent_state(alpha, cutoff):
    r"""The coherent state :math:`\lvert\alpha\rangle`, truncated to a finite Fock basis.

    .. math::

        \lvert\alpha\rangle = e^{-\lvert\alpha\rvert^2/2}
            \sum_{n=0}^{\infty} \frac{\alpha^n}{\sqrt{n!}}\,\lvert n\rangle

    The sum is truncated to ``cutoff`` terms and the result renormalized,
    so it is a good approximation only while :math:`\lvert\alpha\rvert^2
    \ll` ``cutoff`` (i.e. the mean photon number is well within the
    truncated space). Amplitudes are built iteratively in log-space via
    :func:`scipy.special.gammaln` to avoid overflow for large ``cutoff``.

    Parameters
    ----------
    alpha : complex
        Coherent-state amplitude.
    cutoff : int
        Dimension of the truncated Fock space.

    Returns
    -------
    ndarray of shape (cutoff,)
        Complex, L2-normalized state vector.

    Examples
    --------
    >>> psi = coherent_state(0.0, 8)
    >>> psi
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """
    alpha = complex(alpha)
    n = np.arange(cutoff)
    r = abs(alpha)
    if r == 0.0:
        log_amp = np.full(cutoff, -np.inf)
        log_amp[0] = 0.0
    else:
        log_amp = -0.5 * r**2 + n * np.log(r) - 0.5 * gammaln(n + 1.0)
    amp = np.exp(log_amp)
    phase = np.exp(1j * n * np.angle(alpha)) if alpha != 0 else np.ones(cutoff)
    psi = (amp * phase).astype(complex)
    psi /= np.linalg.norm(psi)
    return psi


def squeezed_state(xi, alpha=0.0, cutoff=30):
    r"""The displaced squeezed vacuum state, using the squeeze-then-displace convention.

    Constructs

    .. math::

        \lvert\xi, \alpha\rangle = \hat D(\alpha)\, \hat S(\xi)\, \lvert 0\rangle,
        \qquad
        \hat S(\xi) = \exp\!\left[\frac{\xi^* \hat a^2 - \xi \hat a^{\dagger 2}}{2}\right],
        \qquad
        \hat D(\alpha) = \exp\!\left(\alpha \hat a^\dagger - \alpha^* \hat a\right)

    i.e. the vacuum is squeezed first and then displaced. (The opposite
    order, displace-then-squeeze, gives a different, also physically valid,
    state -- squeeze-then-displace is the convention adopted here and is
    the more common one in the literature.) The ladder operators are built
    on the same truncated Fock basis, so this is only accurate while the
    squeezed/displaced amplitude is small relative to ``cutoff``.

    Parameters
    ----------
    xi : complex
        Squeezing parameter :math:`\xi = r e^{i\theta}`.
    alpha : complex, default=0.0
        Displacement amplitude.
    cutoff : int, default=30
        Dimension of the truncated Fock space.

    Returns
    -------
    ndarray of shape (cutoff,)
        Complex, L2-normalized state vector.

    Examples
    --------
    >>> psi = squeezed_state(0.0, alpha=0.0, cutoff=6)
    >>> psi
    array([1.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j, 0.+0.j])
    """
    xi = complex(xi)
    alpha = complex(alpha)
    a = annihilation_operator(cutoff)
    a_dag = creation_operator(cutoff)
    S = expm(0.5 * (np.conj(xi) * (a @ a) - xi * (a_dag @ a_dag)))
    D = expm(alpha * a_dag - np.conj(alpha) * a)
    vac = fock_state(0, cutoff)
    psi = D @ (S @ vac)
    psi = psi / np.linalg.norm(psi)
    return psi


def _wigner_kernel(n, m, x, p):
    r"""Fock-basis Wigner-function kernel :math:`K_{nm}(x,p)` for :math:`n \le m`.

    Defined so that :math:`W(x,p) = \sum_{n \le m}\left[\rho_{nm} K_{nm}
    + \rho_{mn} K_{nm}^*\right]` (with the ``n == m`` term counted once),
    reducing at :math:`n=m` to the well-known diagonal Fock-state Wigner
    function :math:`W_n(x,p) = (-1)^n/\pi\, e^{-(x^2+p^2)} L_n[2(x^2+p^2)]`.
    """
    s = x**2 + p**2
    z = x - 1j * p
    log_coeff = 0.5 * (gammaln(n + 1.0) - gammaln(m + 1.0))
    coeff = (-1.0) ** n * np.exp(log_coeff)
    return (1.0 / np.pi) * coeff * (np.sqrt(2.0) * z) ** (m - n) * np.exp(-s) * eval_genlaguerre(n, m - n, 2.0 * s)


def compute_wigner_function(state_vector, x_grid, p_grid):
    r"""Wigner quasi-probability distribution of a Fock-basis state.

    Computed from the density matrix :math:`\rho = \lvert\psi\rangle\langle\psi\rvert`
    and the closed-form Fock-basis Wigner matrix elements,

    .. math::

        W(x,p) = \sum_{n,m} \rho_{nm}\, K_{nm}(x,p),

    with :math:`K_{nm}` built from associated Laguerre polynomials of
    :math:`2(x^2+p^2)` (see :func:`scipy.special.eval_genlaguerre`). Uses
    dimensionless quadratures :math:`x = (a+a^\dagger)/\sqrt2`,
    :math:`p = (a-a^\dagger)/(i\sqrt2)`, for which
    :math:`\iint W(x,p)\,dx\,dp = 1` for any normalized state.

    Parameters
    ----------
    state_vector : ndarray of shape (cutoff,)
        Fock-basis ket.
    x_grid : ndarray of shape (Nx,)
        Grid of :math:`x` quadrature values.
    p_grid : ndarray of shape (Np,)
        Grid of :math:`p` quadrature values.

    Returns
    -------
    ndarray of shape (Nx, Np)
        Real-valued Wigner function :math:`W(x,p)`.

    Examples
    --------
    >>> import numpy as np
    >>> vac = fock_state(0, 8)
    >>> xg = np.linspace(-4, 4, 41)
    >>> pg = np.linspace(-4, 4, 41)
    >>> W = compute_wigner_function(vac, xg, pg)
    >>> W.shape
    (41, 41)
    >>> round(float(W[20, 20] * np.pi), 6)
    1.0
    """
    psi = np.asarray(state_vector, dtype=complex).reshape(-1)
    cutoff = psi.shape[0]
    rho = np.outer(psi, psi.conj())
    x_grid = np.asarray(x_grid, dtype=float)
    p_grid = np.asarray(p_grid, dtype=float)
    X, P = np.meshgrid(x_grid, p_grid, indexing="ij")
    W = np.zeros_like(X, dtype=complex)
    for n in range(cutoff):
        for m in range(n, cutoff):
            K = _wigner_kernel(n, m, X, P)
            if m == n:
                W = W + rho[n, n] * K
            else:
                W = W + rho[n, m] * K + rho[m, n] * np.conj(K)
    return W.real


def wigner_negativity(W, x_grid, p_grid):
    r"""Wigner negativity: the integrated negative volume of a Wigner function.

    .. math::

        N_W = \iint \max(0, -W(x,p))\; dx\, dp

    A standard non-classicality diagnostic: zero for classical/Gaussian
    states (coherent, thermal, squeezed vacuum), strictly positive for
    genuinely non-classical states such as Fock states with :math:`n \ge 1`.

    Parameters
    ----------
    W : ndarray of shape (Nx, Np)
        Wigner function, e.g. from :func:`compute_wigner_function`.
    x_grid : ndarray of shape (Nx,)
        Grid of :math:`x` quadrature values.
    p_grid : ndarray of shape (Np,)
        Grid of :math:`p` quadrature values.

    Returns
    -------
    float
        The (non-negative) integrated negative volume.

    Examples
    --------
    >>> import numpy as np
    >>> vac = fock_state(0, 8)
    >>> xg = np.linspace(-5, 5, 61)
    >>> pg = np.linspace(-5, 5, 61)
    >>> W = compute_wigner_function(vac, xg, pg)
    >>> wigner_negativity(W, xg, pg)
    0.0
    """
    negative_part = np.where(W < 0.0, -W, 0.0)
    integral_over_p = trapz(negative_part, p_grid, axis=1)
    return float(trapz(integral_over_p, x_grid))


class JaynesCummingsModel:
    r"""The Jaynes-Cummings model: a two-level atom coupled to a single quantized cavity mode.

    In the rotating-wave approximation, the Hamiltonian is

    .. math::

        \hat H = \omega_c\, \hat a^\dagger \hat a \otimes \hat I
            + \frac{\omega_a}{2}\, \hat I \otimes \hat\sigma_z
            + g\left(\hat a \otimes \hat\sigma_+ + \hat a^\dagger \otimes \hat\sigma_-\right)

    with the cavity (dimension ``cutoff``) as the *left* tensor factor and
    the atom (dimension 2, basis :math:`\{\lvert e\rangle, \lvert g\rangle\}`,
    so :math:`\hat\sigma_z = \mathrm{diag}(+1,-1)`, :math:`\hat\sigma_+ =
    \lvert e\rangle\langle g\rvert`, :math:`\hat\sigma_- = \lvert
    g\rangle\langle e\rvert`) as the *right* factor, i.e. ``H = kron(H_cav,
    H_atom)`` terms throughout. A combined basis state :math:`\lvert n,
    e/g\rangle` therefore sits at flat index ``2*n`` (excited) or ``2*n+1``
    (ground) of the ``2*cutoff``-dimensional Hilbert space.

    Parameters
    ----------
    omega_c : float
        Cavity mode frequency.
    omega_a : float
        Atomic transition frequency.
    g : float
        Atom-cavity coupling strength.
    cutoff : int, default=10
        Truncation of the cavity Fock space.
    """

    def __init__(self, omega_c, omega_a, g, cutoff=10):
        self.omega_c = omega_c
        self.omega_a = omega_a
        self.g = g
        self.cutoff = cutoff

    def hamiltonian(self):
        r"""The Jaynes-Cummings Hamiltonian as a dense matrix.

        Returns
        -------
        ndarray of shape (2*cutoff, 2*cutoff)
            The Hamiltonian :math:`\hat H`, in the ``cavity (x) atom``
            basis ordering documented on the class.

        Examples
        --------
        >>> import numpy as np
        >>> jc = JaynesCummingsModel(omega_c=1.0, omega_a=1.0, g=0.1, cutoff=3)
        >>> H = jc.hamiltonian()
        >>> H.shape
        (6, 6)
        >>> bool(np.allclose(H, H.conj().T))
        True
        """
        cutoff = self.cutoff
        a = annihilation_operator(cutoff)
        a_dag = creation_operator(cutoff)
        N = a_dag @ a
        I2 = np.eye(2, dtype=complex)
        I_cav = np.eye(cutoff, dtype=complex)
        sigma_z = np.diag([1.0, -1.0]).astype(complex)
        sigma_plus = np.array([[0.0, 1.0], [0.0, 0.0]], dtype=complex)  # |e><g|
        sigma_minus = np.array([[0.0, 0.0], [1.0, 0.0]], dtype=complex)  # |g><e|
        H = self.omega_c * np.kron(N, I2) + 0.5 * self.omega_a * np.kron(I_cav, sigma_z) + self.g * (np.kron(a, sigma_plus) + np.kron(a_dag, sigma_minus))
        return H

    def evolve(self, psi0, t_array):
        r"""Time-evolve a state under the Jaynes-Cummings Hamiltonian.

        Diagonalizes :math:`\hat H` once and reconstructs
        :math:`\lvert\psi(t)\rangle = e^{-i\hat H t}\lvert\psi_0\rangle` at
        every requested time from the eigendecomposition, which is much
        faster than exponentiating :math:`\hat H` separately for each
        ``t``.

        Parameters
        ----------
        psi0 : ndarray of shape (2*cutoff,)
            Initial state.
        t_array : ndarray of shape (T,)
            Times at which to evaluate the evolved state.

        Returns
        -------
        ndarray of shape (T, 2*cutoff)
            Complex states :math:`\lvert\psi(t)\rangle` at each requested time.
        """
        H = self.hamiltonian()
        eigvals, eigvecs = np.linalg.eigh(H)
        psi0 = np.asarray(psi0, dtype=complex).reshape(-1)
        c = eigvecs.conj().T @ psi0
        t_array = np.asarray(t_array, dtype=float)
        phase = np.exp(-1j * np.outer(t_array, eigvals))
        coeffs_t = phase * c[np.newaxis, :]
        return coeffs_t @ eigvecs.T

    def excited_state_population(self, t_array, n_photons=0):
        r"""Atomic excited-state population :math:`P_e(t)` starting from :math:`\lvert e, n\rangle`.

        Starts from the initial state :math:`\lvert e, n_\text{photons}\rangle`
        (atom excited, ``n_photons`` photons in the cavity) and returns

        .. math::

            P_e(t) = \sum_n \left\lvert \langle e, n \rvert \psi(t)\rangle \right\rvert^2.

        On resonance (:math:`\omega_c = \omega_a`) with ``n_photons=0``,
        this reduces to the textbook vacuum Rabi formula :math:`P_e(t) =
        \cos^2(gt)`.

        Parameters
        ----------
        t_array : ndarray of shape (T,)
            Times at which to evaluate the population.
        n_photons : int, default=0
            Initial photon number (with the atom excited).

        Returns
        -------
        ndarray of shape (T,)
            Real-valued excited-state population at each time.

        Examples
        --------
        >>> import numpy as np
        >>> jc = JaynesCummingsModel(omega_c=1.0, omega_a=1.0, g=0.5, cutoff=10)
        >>> t = np.array([0.0, np.pi / 2 / 0.5])
        >>> Pe = jc.excited_state_population(t, n_photons=0)
        >>> np.round(Pe, 6)
        array([1., 0.])
        """
        psi0 = fock_state(2 * n_photons, 2 * self.cutoff)
        psi_t = self.evolve(psi0, t_array)
        excited_amplitudes = psi_t[:, 0::2]
        return np.sum(np.abs(excited_amplitudes) ** 2, axis=1).real
