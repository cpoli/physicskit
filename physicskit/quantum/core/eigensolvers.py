r"""Matrix Numerov eigensolver for the time-independent Schrodinger equation.

Solves

.. math::

    -\frac{\hbar^2}{2m}\,\psi''(x) + V(x)\,\psi(x) = E\,\psi(x)

on a uniform grid with :math:`\psi = 0` at the domain boundary, using the
generalized-eigenvalue formulation of the Numerov method (Pillai, Goglio &
Walker, *Am. J. Phys.* **80**, 1017 (2012)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.linalg import eigh
from scipy.sparse import diags

from .._compat import trapz

__all__ = [
    "EigenResult",
    "NumerovSolver",
    "infinite_well",
    "finite_well",
    "asymmetric_step_well",
    "linear_gravitational_well",
    "double_well",
    "harmonic_well",
]


@njit(cache=True)
def _tridiag_arrays(V: np.ndarray, h: float, hbar: float, m: float):
    n = V.shape[0]
    a_main = np.empty(n)
    a_off = np.empty(n - 1)
    b_main = np.empty(n)
    b_off = np.empty(n - 1)

    kin = hbar**2 / (2.0 * m * h * h)
    for i in range(n):
        a_main[i] = 2.0 * kin + (10.0 / 12.0) * V[i]
        b_main[i] = 10.0 / 12.0
    for i in range(n - 1):
        a_off[i] = -kin + (1.0 / 12.0) * 0.5 * (V[i] + V[i + 1])
        b_off[i] = 1.0 / 12.0
    return a_main, a_off, b_main, b_off


@dataclass
class EigenResult:
    """Container for the output of :meth:`NumerovSolver.solve`."""

    x: np.ndarray
    """numpy.ndarray: The spatial grid the wavefunctions are sampled on."""

    energies: np.ndarray
    r"""numpy.ndarray: Eigenenergies :math:`E_n`, ascending, shape ``(n_states,)``."""

    wavefunctions: np.ndarray
    r"""numpy.ndarray: Normalized eigenfunctions :math:`\psi_n(x)`, shape
    ``(n_states, len(x))``, with :math:`\int \lvert\psi_n\rvert^2\,dx = 1`
    and :math:`\psi_n(x[0]) = \psi_n(x[-1]) = 0`."""


class NumerovSolver:
    r"""Bound-state solver for an arbitrary 1D potential :math:`V(x)` on a fixed grid.

    Parameters
    ----------
    x : array_like
        Uniform spatial grid. The wavefunction is forced to vanish at
        ``x[0]`` and ``x[-1]`` (hard walls), so the domain should extend
        well into the classically forbidden region for soft potentials.
    V : callable or array_like
        Potential energy, either a function ``V(x)`` or a precomputed array
        matching ``x``.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.

    Notes
    -----
    The Numerov finite-difference relation for :math:`\psi'' = f\psi` is
    recast as the generalized eigenvalue problem

    .. math::

        \hat A \boldsymbol\psi = E \hat B \boldsymbol\psi, \qquad
        \hat A = -\frac{\hbar^2}{2m h^2}\hat T + \hat B\,\mathrm{diag}(V),

    where :math:`\hat T` and :math:`\hat B` are the tridiagonal matrices
    :math:`\mathrm{tridiag}(1,-2,1)` and :math:`\mathrm{tridiag}(1,10,1)/12`
    and :math:`h` is the grid spacing. Validated against the analytic
    infinite-well and harmonic-oscillator spectra to four decimal places.
    """

    def __init__(self, x: np.ndarray, V, hbar: float = 1.0, m: float = 1.0):
        self.x = np.asarray(x, dtype=float)
        h = self.x[1] - self.x[0]
        if not np.allclose(np.diff(self.x), h, rtol=1e-8):
            raise ValueError("NumerovSolver requires a uniform grid.")
        self.h = h
        self.hbar = hbar
        self.m = m
        self.V_full = np.asarray(V(self.x) if callable(V) else V, dtype=float)

    def solve(self, n_states: int = 6) -> EigenResult:
        """Compute the lowest bound states.

        Parameters
        ----------
        n_states : int, default=6
            Number of lowest-energy eigenstates to return. Must be smaller
            than the number of interior grid points.

        Returns
        -------
        EigenResult
            The energies and normalized wavefunctions.

        Raises
        ------
        ValueError
            If ``n_states`` is not smaller than the number of interior grid
            points.
        """
        # Interior points only; psi vanishes at the two boundary points.
        V_int = self.V_full[1:-1]
        n = V_int.shape[0]
        if n_states >= n:
            raise ValueError("n_states must be smaller than the number of interior grid points.")

        a_main, a_off, b_main, b_off = _tridiag_arrays(V_int, self.h, self.hbar, self.m)

        A = diags([a_off, a_main, a_off], offsets=[-1, 0, 1]).toarray()
        B = diags([b_off, b_main, b_off], offsets=[-1, 0, 1]).toarray()

        eigvals, eigvecs = eigh(A, B)

        energies = eigvals[:n_states]
        psis = np.zeros((n_states, self.x.shape[0]))
        for k in range(n_states):
            psi_int = eigvecs[:, k]
            psi = np.zeros(self.x.shape[0])
            psi[1:-1] = psi_int
            norm = np.sqrt(trapz(psi**2, self.x))
            psis[k] = psi / norm

        return EigenResult(x=self.x, energies=energies, wavefunctions=psis)


# --- Convenience potential builders ---------------------------------------


def infinite_well():
    r"""The infinite square well: :math:`V(x) = 0` inside the domain.

    The grid boundaries passed to :class:`NumerovSolver` act as the
    infinite walls, since the wavefunction is forced to vanish there.

    Returns
    -------
    callable
        A function ``V(x)`` returning zeros the same shape as ``x``.
    """

    def V(x):
        return np.zeros_like(x)

    return V


def finite_well(V0: float, width: float):
    r"""A finite square well,

    .. math::

        V(x) = \begin{cases} 0 & \lvert x\rvert \le \text{width}/2 \\
                              V_0 & \text{otherwise} \end{cases}.

    Parameters
    ----------
    V0 : float
        Height of the barrier outside the well.
    width : float
        Full width of the well (zero-potential region).

    Returns
    -------
    callable
        A function ``V(x)``.
    """

    def V(x):
        return np.where(np.abs(x) <= width / 2, 0.0, V0)

    return V


def asymmetric_step_well(width: float, V_left: float, V_right: float):
    r"""A well of unequal left/right wall heights,

    .. math::

        V(x) = \begin{cases}
            V_\text{left}  & x < -\text{width}/2 \\
            0              & \lvert x\rvert \le \text{width}/2 \\
            V_\text{right} & x > \text{width}/2
        \end{cases}.

    Parameters
    ----------
    width : float
        Full width of the zero-potential region.
    V_left : float
        Wall height on the left side.
    V_right : float
        Wall height on the right side.

    Returns
    -------
    callable
        A function ``V(x)``.
    """

    def V(x):
        out = np.zeros_like(x)
        out[x < -width / 2] = V_left
        out[x > width / 2] = V_right
        return out

    return V


def linear_gravitational_well(alpha: float):
    r"""The V-shaped 'quantum bouncer' well, :math:`V(x) = \alpha \lvert x\rvert`.

    A particle in this potential is the quantum analogue of a ball bouncing
    under uniform gravity (:math:`\alpha = mg` on the half-line); its
    eigenstates are Airy functions (see :func:`physicskit.quantum.chapters.potentials.airy_bouncer_energies`).

    Parameters
    ----------
    alpha : float
        Slope of the linear potential (e.g. :math:`mg` for gravity).

    Returns
    -------
    callable
        A function ``V(x)``.
    """

    def V(x):
        return alpha * np.abs(x)

    return V


def double_well(lam: float, a: float):
    r"""The symmetric quartic double well, :math:`V(x) = \lambda (x^2 - a^2)^2`.

    Has minima at :math:`x = \pm a`, each of depth 0, separated by a barrier
    of height :math:`\lambda a^4` at :math:`x=0`.

    Parameters
    ----------
    lam : float
        Overall strength :math:`\lambda` of the quartic potential.
    a : float
        Half-separation of the two minima.

    Returns
    -------
    callable
        A function ``V(x)``.
    """

    def V(x):
        return lam * (x**2 - a**2) ** 2

    return V


def harmonic_well(m: float = 1.0, omega: float = 1.0):
    r"""The quantum harmonic oscillator potential,
    :math:`V(x) = \tfrac{1}{2} m \omega^2 x^2`.

    Parameters
    ----------
    m : float, default=1.0
        Particle mass.
    omega : float, default=1.0
        Angular frequency :math:`\omega`.

    Returns
    -------
    callable
        A function ``V(x)``.
    """

    def V(x):
        return 0.5 * m * omega**2 * x**2

    return V
