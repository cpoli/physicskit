r"""Dirac algebra helpers: Pauli matrices, spin/ladder operators, commutators.

Conventions follow Cohen-Tannoudji, Diu & Laloe, with :math:`\hbar = 1`
unless a function explicitly takes ``hbar`` as a parameter.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "sigma_x",
    "sigma_y",
    "sigma_z",
    "identity2",
    "PAULI",
    "spin_operator",
    "annihilation_operator",
    "creation_operator",
    "number_operator",
    "position_operator",
    "momentum_operator",
    "commutator",
    "anticommutator",
    "expectation",
    "is_hermitian",
]

# --- Pauli matrices -------------------------------------------------------

sigma_x = np.array([[0, 1], [1, 0]], dtype=complex)
r"""The Pauli matrix :math:`\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}`."""

sigma_y = np.array([[0, -1j], [1j, 0]], dtype=complex)
r"""The Pauli matrix :math:`\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}`."""

sigma_z = np.array([[1, 0], [0, -1]], dtype=complex)
r"""The Pauli matrix :math:`\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}`."""

identity2 = np.eye(2, dtype=complex)
"""The :math:`2\\times 2` identity matrix."""

PAULI = {"x": sigma_x, "y": sigma_y, "z": sigma_z, "i": identity2}
"""Dict mapping ``'x'``, ``'y'``, ``'z'``, ``'i'`` to the corresponding matrix."""


def spin_operator(axis: str, s: float = 0.5, hbar: float = 1.0) -> np.ndarray:
    r"""Angular-momentum operator :math:`\hat S_\text{axis}` for spin quantum number ``s``.

    For :math:`s=1/2` this reduces to :math:`(\hbar/2)\,\sigma_\text{axis}`. For
    general :math:`s`, the :math:`(2s+1)`-dimensional matrix representation is
    built from the ladder operators

    .. math::

        \hat S_\pm \lvert s, m\rangle
            = \hbar\sqrt{s(s+1) - m(m\pm 1)}\, \lvert s, m\pm 1\rangle,

    with :math:`\hat S_x = (\hat S_+ + \hat S_-)/2` and
    :math:`\hat S_y = (\hat S_+ - \hat S_-)/2i`.

    Parameters
    ----------
    axis : {'x', 'y', 'z', '+', '-'}
        Which component (or ladder operator) to return.
    s : float, default=0.5
        Spin quantum number (may be integer or half-integer).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    numpy.ndarray
        The :math:`(2s+1)\times(2s+1)` operator matrix, in the
        :math:`\lvert s, m\rangle` basis ordered :math:`m = s, s-1, \dots, -s`.

    Raises
    ------
    ValueError
        If ``axis`` is not one of ``'x'``, ``'y'``, ``'z'``, ``'+'``, ``'-'``.
    """
    dim = int(round(2 * s + 1))
    m_values = np.array([s - k for k in range(dim)])  # m = s, s-1, ..., -s

    if axis == "z":
        return hbar * np.diag(m_values).astype(complex)

    s_plus = np.zeros((dim, dim), dtype=complex)
    for k in range(dim - 1):
        m = m_values[k + 1]
        coeff = hbar * np.sqrt(s * (s + 1) - m * (m + 1))
        s_plus[k, k + 1] = coeff
    s_minus = s_plus.conj().T

    if axis == "+":
        return s_plus
    if axis == "-":
        return s_minus
    if axis == "x":
        return 0.5 * (s_plus + s_minus)
    if axis == "y":
        return (s_plus - s_minus) / (2j)
    raise ValueError(f"Unknown axis {axis!r}; expected 'x', 'y', 'z', '+' or '-'.")


# --- Harmonic-oscillator ladder operators (truncated Fock basis) ---------


def annihilation_operator(n_max: int) -> np.ndarray:
    r"""Lowering operator :math:`\hat a` in a truncated Fock basis.

    Acts as :math:`\hat a \lvert n\rangle = \sqrt{n}\,\lvert n-1\rangle` on the
    truncated basis :math:`\{\lvert 0\rangle, \dots, \lvert n_\text{max}-1\rangle\}`.

    Parameters
    ----------
    n_max : int
        Dimension of the truncated Fock space.

    Returns
    -------
    numpy.ndarray
        The :math:`n_\text{max}\times n_\text{max}` matrix representation of
        :math:`\hat a`.
    """
    a = np.zeros((n_max, n_max), dtype=complex)
    for n in range(1, n_max):
        a[n - 1, n] = np.sqrt(n)
    return a


def creation_operator(n_max: int) -> np.ndarray:
    r"""Raising operator :math:`\hat a^\dagger`, the Hermitian conjugate of
    :func:`annihilation_operator`.

    Parameters
    ----------
    n_max : int
        Dimension of the truncated Fock space.

    Returns
    -------
    numpy.ndarray
        The :math:`n_\text{max}\times n_\text{max}` matrix representation of
        :math:`\hat a^\dagger`.
    """
    return annihilation_operator(n_max).conj().T


def number_operator(n_max: int) -> np.ndarray:
    r"""Number operator :math:`\hat N = \hat a^\dagger \hat a`, diagonal with
    eigenvalues :math:`0, 1, \dots, n_\text{max}-1`.

    Parameters
    ----------
    n_max : int
        Dimension of the truncated Fock space.

    Returns
    -------
    numpy.ndarray
        The :math:`n_\text{max}\times n_\text{max}` matrix representation of
        :math:`\hat N`.
    """
    a = annihilation_operator(n_max)
    return a.conj().T @ a


def position_operator(n_max: int, m: float = 1.0, omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    r"""Harmonic-oscillator position operator in the truncated Fock basis,

    .. math::

        \hat x = \sqrt{\frac{\hbar}{2 m \omega}} \left(\hat a + \hat a^\dagger\right).

    Parameters
    ----------
    n_max : int
        Dimension of the truncated Fock space.
    m : float, default=1.0
        Particle mass.
    omega : float, default=1.0
        Angular frequency of the oscillator.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    numpy.ndarray
        The :math:`n_\text{max}\times n_\text{max}` matrix representation of
        :math:`\hat x`.
    """
    a = annihilation_operator(n_max)
    return np.sqrt(hbar / (2 * m * omega)) * (a + a.conj().T)


def momentum_operator(n_max: int, m: float = 1.0, omega: float = 1.0, hbar: float = 1.0) -> np.ndarray:
    r"""Harmonic-oscillator momentum operator in the truncated Fock basis,

    .. math::

        \hat p = i\sqrt{\frac{\hbar m \omega}{2}} \left(\hat a^\dagger - \hat a\right).

    Parameters
    ----------
    n_max : int
        Dimension of the truncated Fock space.
    m : float, default=1.0
        Particle mass.
    omega : float, default=1.0
        Angular frequency of the oscillator.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    numpy.ndarray
        The :math:`n_\text{max}\times n_\text{max}` matrix representation of
        :math:`\hat p`.
    """
    a = annihilation_operator(n_max)
    return 1j * np.sqrt(hbar * m * omega / 2) * (a.conj().T - a)


# --- Algebra utilities ------------------------------------------------------


def commutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""The commutator :math:`[\hat A, \hat B] = \hat A \hat B - \hat B \hat A`.

    Parameters
    ----------
    A, B : numpy.ndarray
        Square matrices of matching shape.

    Returns
    -------
    numpy.ndarray
        :math:`\hat A \hat B - \hat B \hat A`.
    """
    return A @ B - B @ A


def anticommutator(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    r"""The anticommutator :math:`\{\hat A, \hat B\} = \hat A \hat B + \hat B \hat A`.

    Parameters
    ----------
    A, B : numpy.ndarray
        Square matrices of matching shape.

    Returns
    -------
    numpy.ndarray
        :math:`\hat A \hat B + \hat B \hat A`.
    """
    return A @ B + B @ A


def expectation(operator: np.ndarray, state: np.ndarray) -> complex:
    r"""Expectation value :math:`\langle\psi\rvert \hat A \lvert\psi\rangle` of
    an operator in a discrete-basis state vector.

    Parameters
    ----------
    operator : numpy.ndarray
        The operator matrix :math:`\hat A`.
    state : numpy.ndarray
        The (not necessarily normalized) state vector :math:`\lvert\psi\rangle`;
        flattened if not already 1D.

    Returns
    -------
    complex
        :math:`\langle\psi\rvert \hat A \lvert\psi\rangle`.
    """
    state = np.asarray(state).reshape(-1)
    return complex(np.vdot(state, operator @ state))


def is_hermitian(A: np.ndarray, atol: float = 1e-10) -> bool:
    r"""Check whether a matrix is Hermitian, :math:`\hat A = \hat A^\dagger`.

    Parameters
    ----------
    A : numpy.ndarray
        The matrix to test.
    atol : float, default=1e-10
        Absolute tolerance passed to :func:`numpy.allclose`.

    Returns
    -------
    bool
        ``True`` if ``A`` equals its conjugate transpose within ``atol``.
    """
    return np.allclose(A, A.conj().T, atol=atol)
