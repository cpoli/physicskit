r"""Split-operator FFT propagators for the time-dependent Schrodinger equation.

Both solvers implement the symmetric (Strang) splitting

.. math::

    \psi(t+dt) = e^{-i \hat V dt / 2\hbar}\, e^{-i \hat T dt/\hbar}\,
                 e^{-i \hat V dt / 2\hbar}\, \psi(t),

which is unitary to machine precision regardless of :math:`dt` (probability
is exactly conserved up to FFT/round-off error), and second-order accurate
in :math:`dt`.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .._compat import trapz

__all__ = ["SplitOperatorSolver1D", "SplitOperatorSolver2D"]


class SplitOperatorSolver1D:
    r"""Propagate a 1D wavepacket under potential :math:`V(x)` or :math:`V(x,t)`.

    Parameters
    ----------
    x : numpy.ndarray
        Uniform spatial grid.
    V : callable or array_like
        Potential energy: a function ``V(x)`` (static) or ``V(x, t)``
        (time-dependent, detected by inspecting the function's signature),
        or a precomputed array matching ``x``.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    dt : float, default=1e-3
        Time step.
    """

    def __init__(self, x: np.ndarray, V, hbar: float = 1.0, m: float = 1.0, dt: float = 1e-3):
        self.x = np.asarray(x, dtype=float)
        self.N = self.x.shape[0]
        self.dx = self.x[1] - self.x[0]
        self.hbar = hbar
        self.m = m
        self.dt = dt
        self._V = V
        self._time_dependent = callable(V) and _accepts_time(V, n_spatial_args=1)

        self.k = 2 * np.pi * np.fft.fftfreq(self.N, d=self.dx)
        self._kin_phase = np.exp(-1j * hbar * self.k**2 * dt / (2 * m))

        if not self._time_dependent:
            V0 = V(self.x) if callable(V) else np.asarray(V, dtype=float)
            self._half_pot_phase = np.exp(-1j * V0 * dt / (2 * hbar))
        else:
            self._half_pot_phase = None

    def _half_potential_phase(self, t: float) -> np.ndarray:
        if self._half_pot_phase is not None:
            return self._half_pot_phase
        Vt = self._V(self.x, t)
        return np.exp(-1j * Vt * self.dt / (2 * self.hbar))

    def step(self, psi: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Advance the wavefunction by one time step ``dt``.

        Parameters
        ----------
        psi : numpy.ndarray
            Complex wavefunction at time ``t``, sampled on ``self.x``.
        t : float, default=0.0
            Current time (only used if the potential is time-dependent).

        Returns
        -------
        numpy.ndarray
            The wavefunction at time ``t + dt``.
        """
        psi = psi * self._half_potential_phase(t)
        psi = np.fft.ifft(self._kin_phase * np.fft.fft(psi))
        psi = psi * self._half_potential_phase(t + self.dt)
        return psi

    def propagate(self, psi0: np.ndarray, n_steps: int, t0: float = 0.0, save_every: int = 1) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Propagate an initial state and record its history.

        Parameters
        ----------
        psi0 : numpy.ndarray
            Initial wavefunction (renormalized internally).
        n_steps : int
            Total number of time steps to take.
        t0 : float, default=0.0
            Initial time.
        save_every : int, default=1
            Save a snapshot every ``save_every`` steps (plus the initial state).

        Returns
        -------
        frames : numpy.ndarray
            Stacked wavefunction snapshots, shape ``(n_saved, len(x))``.
        times : numpy.ndarray
            The time of each snapshot, shape ``(n_saved,)``.
        """
        psi = np.asarray(psi0, dtype=complex).copy()
        psi /= np.sqrt(trapz(np.abs(psi) ** 2, self.x))

        frames = [psi.copy()]
        times = [t0]
        t = t0
        for step_idx in range(1, n_steps + 1):
            psi = self.step(psi, t)
            t = t0 + step_idx * self.dt
            if step_idx % save_every == 0:
                frames.append(psi.copy())
                times.append(t)
        return np.array(frames), np.array(times)

    def norm(self, psi: np.ndarray) -> float:
        r"""Total probability :math:`\int \lvert\psi(x)\rvert^2\,dx`.

        Parameters
        ----------
        psi : numpy.ndarray
            Wavefunction sampled on ``self.x``.

        Returns
        -------
        float
            The normalization integral (should stay at 1.0 under unitary
            evolution).
        """
        return float(trapz(np.abs(psi) ** 2, self.x))


class SplitOperatorSolver2D:
    r"""Propagate a 2D wavepacket under potential :math:`V(x,y)` or :math:`V(x,y,t)`.

    Parameters
    ----------
    x, y : numpy.ndarray
        Uniform spatial grids along each axis.
    V : callable or array_like
        Potential energy: a function ``V(x, y)`` (static) or ``V(x, y, t)``
        (time-dependent, detected by inspecting the function's signature),
        or a precomputed array matching the ``(x, y)`` meshgrid.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    dt : float, default=1e-3
        Time step.
    """

    def __init__(self, x: np.ndarray, y: np.ndarray, V, hbar: float = 1.0, m: float = 1.0, dt: float = 1e-3):
        self.x = np.asarray(x, dtype=float)
        self.y = np.asarray(y, dtype=float)
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]
        self.hbar = hbar
        self.m = m
        self.dt = dt
        self._V = V
        self._time_dependent = callable(V) and _accepts_time(V, n_spatial_args=2)

        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="ij")
        kx = 2 * np.pi * np.fft.fftfreq(self.x.shape[0], d=self.dx)
        ky = 2 * np.pi * np.fft.fftfreq(self.y.shape[0], d=self.dy)
        KX, KY = np.meshgrid(kx, ky, indexing="ij")
        self._kin_phase = np.exp(-1j * hbar * (KX**2 + KY**2) * dt / (2 * m))

        if not self._time_dependent:
            V0 = V(self.X, self.Y) if callable(V) else np.asarray(V, dtype=float)
            self._half_pot_phase = np.exp(-1j * V0 * dt / (2 * hbar))
        else:
            self._half_pot_phase = None

    def _half_potential_phase(self, t: float) -> np.ndarray:
        if self._half_pot_phase is not None:
            return self._half_pot_phase
        Vt = self._V(self.X, self.Y, t)
        return np.exp(-1j * Vt * self.dt / (2 * self.hbar))

    def step(self, psi: np.ndarray, t: float = 0.0) -> np.ndarray:
        """Advance the wavefunction by one time step ``dt``.

        Parameters
        ----------
        psi : numpy.ndarray
            Complex wavefunction at time ``t``, sampled on the ``(x, y)`` grid.
        t : float, default=0.0
            Current time (only used if the potential is time-dependent).

        Returns
        -------
        numpy.ndarray
            The wavefunction at time ``t + dt``.
        """
        psi = psi * self._half_potential_phase(t)
        psi = np.fft.ifft2(self._kin_phase * np.fft.fft2(psi))
        psi = psi * self._half_potential_phase(t + self.dt)
        return psi

    def propagate(self, psi0: np.ndarray, n_steps: int, t0: float = 0.0, save_every: int = 1):
        """Propagate an initial state and record its history.

        Parameters
        ----------
        psi0 : numpy.ndarray
            Initial wavefunction (renormalized internally).
        n_steps : int
            Total number of time steps to take.
        t0 : float, default=0.0
            Initial time.
        save_every : int, default=1
            Save a snapshot every ``save_every`` steps (plus the initial state).

        Returns
        -------
        frames : numpy.ndarray
            Stacked wavefunction snapshots, shape ``(n_saved, *X.shape)``.
        times : numpy.ndarray
            The time of each snapshot, shape ``(n_saved,)``.
        """
        psi = np.asarray(psi0, dtype=complex).copy()
        psi /= np.sqrt(self.norm(psi))

        frames = [psi.copy()]
        times = [t0]
        t = t0
        for step_idx in range(1, n_steps + 1):
            psi = self.step(psi, t)
            t = t0 + step_idx * self.dt
            if step_idx % save_every == 0:
                frames.append(psi.copy())
                times.append(t)
        return np.array(frames), np.array(times)

    def norm(self, psi: np.ndarray) -> float:
        r"""Total probability :math:`\iint \lvert\psi(x,y)\rvert^2\,dx\,dy`.

        Parameters
        ----------
        psi : numpy.ndarray
            Wavefunction sampled on the ``(x, y)`` grid.

        Returns
        -------
        float
            The normalization integral (should stay at 1.0 under unitary
            evolution).
        """
        density = np.abs(psi) ** 2
        return float(trapz(trapz(density, self.y, axis=1), self.x))


def _accepts_time(func: Callable, n_spatial_args: int) -> bool:
    """True if ``func`` takes more than ``n_spatial_args`` positional
    parameters, i.e. it is V(x, ..., t) rather than V(x, ...)."""
    try:
        import inspect

        return len(inspect.signature(func).parameters) > n_spatial_args
    except (TypeError, ValueError):
        return False
