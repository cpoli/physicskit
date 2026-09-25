"""Quantized versions of physicskit.chaos's classical maps: the kicked rotor and the baker's map.

Both are built the standard way in the quantum-chaos literature: as a
finite-dimensional (``dim``-by-``dim``) unitary "Floquet" operator advancing a
state vector by one map iteration, constructed from the discrete Fourier
transform (DFT) rather than by discretizing a differential equation. This
makes them cheap to build and exactly unitary to machine precision, at the
cost of only being able to represent states in a single Hilbert space
dimension at a time (there is no continuum limit to refine towards, unlike
:class:`~physicskit.chaos.quantum.billiards.QuantumBilliard`); the classical limit is
instead approached by taking ``dim -> infinity`` (equivalently ``hbar -> 0``).

Random-matrix-theory analyses of these maps' spectra (e.g. nearest-neighbor
spacing distributions, spectral rigidity) are intentionally *not* provided
here -- that is the job of the separate ``physicskit.rmt`` package, which can consume
the eigenphases returned by :meth:`QuantumKickedRotor.eigenphases` or
:meth:`QuantumBakersMap.eigenphases` directly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray

from physicskit.chaos.core.base_system import get_init_params
from physicskit.chaos.exceptions import InvalidParameterError
from physicskit.chaos.quantum.husimi import husimi_function


def _dft_matrix(n: int) -> NDArray[np.complex128]:
    """The ``n``-by-``n`` unitary discrete Fourier transform matrix."""
    j = np.arange(n)
    matrix = np.exp(-2j * np.pi * np.outer(j, j) / n) / np.sqrt(n)
    return np.asarray(matrix, dtype=np.complex128)


def _init_repr(obj: object) -> str:
    """Build an ``eval``-able-looking repr from an object's ``__init__`` signature.

    Local copy of :func:`physicskit.chaos.core.base_system._init_repr` (kept private
    there); see :func:`~physicskit.chaos.core.base_system.get_init_params`.
    """
    parts = [f"{name}={value!r}" for name, value in get_init_params(obj).items()]
    return f"{type(obj).__name__}({', '.join(parts)})"


class QuantumKickedRotor:
    """The quantum kicked rotor: the standard map's quantization.

    Built as the one-period Floquet operator of a rotor periodically kicked
    by a potential ``k * cos(theta)``, in exact correspondence with
    physicskit.chaos's classical :class:`~physicskit.chaos.systems.maps.StandardMap`
    (``p_new = p + k*sin(theta)``, ``theta_new = theta + p_new``): the
    Hilbert space is the `dim`-point position (angle) representation on
    ``[0, 2*pi)``, and the Floquet operator alternates a kick phase (diagonal
    in the angle basis) with a free-rotation phase (diagonal in the momentum
    basis), transforming between the two via the discrete Fourier transform.

    As `dim` grows (equivalently, as `hbar` shrinks towards its default
    ``2*pi/dim``), the quantum dynamics of a narrow wavepacket increasingly
    tracks the corresponding classical
    :class:`~physicskit.chaos.systems.maps.StandardMap` orbit, until the packet
    spreads across a chaotic region -- the quantum-classical correspondence
    breaking down being one of the central phenomena of quantum chaos.

    Parameters
    ----------
    k : float, default 1.0
        Kick strength; matches the classical
        :class:`~physicskit.chaos.systems.maps.StandardMap`'s `k` exactly.
    dim : int, default 64
        Hilbert space dimension (number of angle basis states).
    hbar : float, optional
        Effective Planck constant; defaults to ``2*pi/dim``, the standard
        choice that keeps the quantized torus's phase-space cell count equal
        to `dim`.

    Attributes
    ----------
    k : float
        Kick strength.
    dim : int
        Hilbert space dimension.
    hbar : float
        Effective Planck constant.

    Raises
    ------
    InvalidParameterError
        If `dim` is smaller than 2.
    """

    def __init__(self, k: float = 1.0, dim: int = 64, hbar: float | None = None):
        if dim < 2:
            raise InvalidParameterError("dim must be at least 2")
        self.k = float(k)
        self.dim = int(dim)
        self.hbar = float(hbar) if hbar is not None else 2.0 * np.pi / self.dim

    def floquet_operator(self) -> NDArray[np.complex128]:
        """Build the one-period Floquet (evolution) operator.

        Returns
        -------
        ndarray of complex, shape (dim, dim)
            Unitary matrix advancing a state, in the angle representation,
            by one kick-and-rotation period.
        """
        n = self.dim
        theta = 2.0 * np.pi * np.arange(n) / n
        momentum = np.arange(n)
        kick_phase = np.exp(-1j * (self.k / self.hbar) * np.cos(theta))
        kinetic_phase = np.exp(-1j * self.hbar * momentum**2 / 2.0)
        dft = _dft_matrix(n)
        result = dft.conj().T @ (kinetic_phase[:, None] * dft) @ np.diag(kick_phase)
        return np.asarray(result, dtype=np.complex128)

    def evolve(self, psi: ArrayLike, n_steps: int = 1) -> NDArray[np.complex128]:
        """Propagate a state through ``n_steps`` kicks.

        Parameters
        ----------
        psi : array_like of complex, shape (dim,)
            Initial state in the angle representation; normalized
            internally.
        n_steps : int, default 1
            Number of Floquet periods to advance.

        Returns
        -------
        ndarray of complex, shape (n_steps + 1, dim)
            The state after each kick, including the (normalized) initial
            state as row 0.
        """
        u = self.floquet_operator()
        state = np.asarray(psi, dtype=np.complex128)
        state = state / np.linalg.norm(state)
        states = np.empty((n_steps + 1, self.dim), dtype=np.complex128)
        states[0] = state
        for i in range(n_steps):
            state = u @ state
            states[i + 1] = state
        return states

    def eigenphases(self) -> NDArray[np.float64]:
        """Quasi-energies (eigenphases) of the Floquet operator.

        Hand these to ``physicskit.rmt`` to study their spacing statistics.

        Returns
        -------
        ndarray of float, shape (dim,)
            Eigenphases (angles of the Floquet operator's unit-modulus
            eigenvalues), sorted ascending, in radians in ``(-pi, pi]``.
        """
        eigvals = np.linalg.eigvals(self.floquet_operator())
        return np.sort(np.angle(eigvals))

    def coherent_state(self, theta0: float, p0: float) -> NDArray[np.complex128]:
        """A minimum-uncertainty wavepacket centered at ``(theta0, p0)``.

        Useful as a semiclassical initial state for :meth:`evolve`, to watch
        the quantum dynamics track (and eventually depart from) the
        corresponding classical orbit.

        Parameters
        ----------
        theta0, p0 : float
            Phase-space center of the wavepacket.

        Returns
        -------
        ndarray of complex, shape (dim,)
            Normalized coherent state in the angle representation.
        """
        n = self.dim
        theta = 2.0 * np.pi * np.arange(n) / n
        sigma2 = self.hbar
        psi = np.zeros(n, dtype=np.complex128)
        for m in range(-3, 4):
            dtheta = theta - theta0 - 2.0 * np.pi * m
            psi += np.exp(-(dtheta**2) / (2.0 * sigma2)) * np.exp(1j * p0 * dtheta / self.hbar)
        return psi / np.linalg.norm(psi)

    def husimi(self, psi: ArrayLike, resolution: int = 80, n_wraps: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Husimi phase-space distribution of a state; see :func:`~physicskit.chaos.quantum.husimi.husimi_function`.

        Parameters
        ----------
        psi : array_like of complex, shape (dim,)
            State in the angle representation.
        resolution : int, default 80
            Number of grid points along each of the ``theta`` and ``p`` axes.
        n_wraps : int, default 3
            Number of periodic images summed to periodize the coherent
            states used.

        Returns
        -------
        Theta, P, husimi : ndarray of float, shape (resolution, resolution)
            Phase-space grid and the (peak-normalized) Husimi distribution.
        """
        state = np.asarray(psi, dtype=np.complex128)
        return husimi_function(state, self.hbar, 2.0 * np.pi, resolution=resolution, n_wraps=n_wraps)

    def __repr__(self) -> str:
        return _init_repr(self)


class QuantumBakersMap:
    """The quantum baker's map: the (generalized) baker's map's quantization.

    Built via the Balazs-Voros/Saraceno construction, generalized to an
    arbitrary cut `alpha` in exact correspondence with physicskit.chaos's classical
    :class:`~physicskit.chaos.systems.maps.BakersMap`: the Floquet operator applies
    the discrete Fourier transform separately to the ``q < alpha`` and
    ``q >= alpha`` position-basis blocks (mirroring the classical map
    stretching each piece independently), then transforms the result back to
    the full position representation.

    Parameters
    ----------
    dim : int
        Hilbert space dimension (number of position basis states).
    alpha : float, default 0.5
        Cut position, in ``(0, 1)``; matches the classical
        :class:`~physicskit.chaos.systems.maps.BakersMap`'s `alpha`. The classic
        Balazs-Voros construction is the ``alpha=0.5`` case with `dim` even.

    Attributes
    ----------
    dim : int
        Hilbert space dimension.
    alpha : float
        Cut position.
    hbar : float
        Effective reduced Planck constant, ``1 / (2*pi*dim)``: the unit
        torus holds ``dim = 1/h = 1/(2*pi*hbar)`` states (Balazs & Voros
        1989), so plane waves are ``exp(i*p*q/hbar) = exp(2*pi*i*dim*p*q)``.

    Raises
    ------
    InvalidParameterError
        If `alpha` does not satisfy ``0 < alpha < 1``, or if rounding
        ``alpha * dim`` to the nearest integer would leave either of the two
        blocks with fewer than 1 basis state.

    Notes
    -----
    Exact correspondence with the classical cut requires ``alpha * dim`` to
    be an integer; for other values, the nearest integer split is used, and
    :attr:`alpha` is left at the value the caller requested rather than
    silently adjusted to the value actually realized -- pass a `dim` that
    makes ``alpha * dim`` (near-)integral for the closest match.
    """

    def __init__(self, dim: int, alpha: float = 0.5):
        if not 0.0 < alpha < 1.0:
            raise InvalidParameterError("alpha must satisfy 0 < alpha < 1")
        n1 = round(alpha * dim)
        if n1 < 1 or n1 > dim - 1:
            raise InvalidParameterError("alpha * dim must round to an integer strictly between 0 and dim")
        self.dim = int(dim)
        self.alpha = float(alpha)
        self.hbar = 1.0 / (2.0 * np.pi * self.dim)
        self._n1 = n1
        self._n2 = dim - n1

    def floquet_operator(self) -> NDArray[np.complex128]:
        """Build the one-iteration Floquet (evolution) operator.

        Returns
        -------
        ndarray of complex, shape (dim, dim)
            Unitary matrix advancing a state, in the position
            representation, by one map iteration.
        """
        n = self.dim
        block = np.zeros((n, n), dtype=np.complex128)
        block[: self._n1, : self._n1] = _dft_matrix(self._n1)
        block[self._n1 :, self._n1 :] = _dft_matrix(self._n2)
        return _dft_matrix(n).conj().T @ block

    def evolve(self, psi: ArrayLike, n_steps: int = 1) -> NDArray[np.complex128]:
        """Propagate a state through ``n_steps`` map iterations.

        Parameters
        ----------
        psi : array_like of complex, shape (dim,)
            Initial state in the position representation; normalized
            internally.
        n_steps : int, default 1
            Number of map iterations to advance.

        Returns
        -------
        ndarray of complex, shape (n_steps + 1, dim)
            The state after each iteration, including the (normalized)
            initial state as row 0.
        """
        u = self.floquet_operator()
        state = np.asarray(psi, dtype=np.complex128)
        state = state / np.linalg.norm(state)
        states = np.empty((n_steps + 1, self.dim), dtype=np.complex128)
        states[0] = state
        for i in range(n_steps):
            state = u @ state
            states[i + 1] = state
        return states

    def eigenphases(self) -> NDArray[np.float64]:
        """Quasi-energies (eigenphases) of the Floquet operator.

        Hand these to ``physicskit.rmt`` to study their spacing statistics.

        Returns
        -------
        ndarray of float, shape (dim,)
            Eigenphases, sorted ascending, in radians in ``(-pi, pi]``.
        """
        eigvals = np.linalg.eigvals(self.floquet_operator())
        return np.sort(np.angle(eigvals))

    def coherent_state(self, q0: float, p0: float) -> NDArray[np.complex128]:
        """A minimum-uncertainty wavepacket centered at ``(q0, p0)``.

        Parameters
        ----------
        q0, p0 : float
            Phase-space center of the wavepacket, each in ``[0, 1)``.

        Returns
        -------
        ndarray of complex, shape (dim,)
            Normalized coherent state in the position representation.
        """
        n = self.dim
        q = np.arange(n) / n
        sigma2 = self.hbar
        psi = np.zeros(n, dtype=np.complex128)
        for m in range(-3, 4):
            dq = q - q0 - m
            psi += np.exp(-(dq**2) / (2.0 * sigma2)) * np.exp(1j * p0 * dq / self.hbar)
        return psi / np.linalg.norm(psi)

    def husimi(self, psi: ArrayLike, resolution: int = 80, n_wraps: int = 3) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """Husimi phase-space distribution of a state; see :func:`~physicskit.chaos.quantum.husimi.husimi_function`.

        Parameters
        ----------
        psi : array_like of complex, shape (dim,)
            State in the position representation.
        resolution : int, default 80
            Number of grid points along each of the ``q`` and ``p`` axes.
        n_wraps : int, default 3
            Number of periodic images summed to periodize the coherent
            states used.

        Returns
        -------
        Q, P, husimi : ndarray of float, shape (resolution, resolution)
            Phase-space grid and the (peak-normalized) Husimi distribution.
        """
        state = np.asarray(psi, dtype=np.complex128)
        return husimi_function(state, self.hbar, 1.0, resolution=resolution, n_wraps=n_wraps)

    def __repr__(self) -> str:
        return _init_repr(self)
