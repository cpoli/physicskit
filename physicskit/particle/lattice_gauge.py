r"""Compact U(1) lattice gauge theory: Monte Carlo, Wilson loops, and the confining area law.

Wilson's lattice formulation (Phys. Rev. D 10, 2445 (1974)) places the
gauge field on the links of a hypercubic lattice as phases
:math:`U_\mu(x) = e^{i\theta_\mu(x)}`, with action

.. math::

    S = \beta\sum_P \left(1 - \cos\theta_P\right), \qquad
    \theta_P = \theta_\mu(x) + \theta_\nu(x + \hat\mu) - \theta_\mu(x + \hat\nu) - \theta_\nu(x),

summed over all plaquettes. The Wilson loop :math:`W(R, T) = \langle
\cos\sum_{\ell\in C}\theta_\ell\rangle` of an :math:`R\times T`
rectangle measures the static potential, :math:`W \sim e^{-V(R)T}`; an
*area law* :math:`W \sim e^{-\sigma RT}` means a linear potential
:math:`V = \sigma R`, i.e. confinement. In two dimensions compact U(1)
confines at every coupling and is exactly solvable (e.g. Gross and
Witten, Phys. Rev. D 21, 446 (1980)).

Dimensionless lattice units (lattice spacing :math:`a = 1`). Link updates
use the Metropolis algorithm in ``@njit`` sweeps over precomputed
neighbor tables.

- :class:`U1LatticeGauge` -- the Monte Carlo simulation in :math:`d = 2`
  or :math:`3` with periodic boundaries.
- :func:`u1_2d_wilson_loop_exact`, :func:`u1_2d_string_tension` -- the
  exact 2D results.
- :func:`creutz_ratio` -- the string-tension estimator of Creutz (Phys.
  Rev. D 21, 2308 (1980)).
"""

from __future__ import annotations

import numpy as np
from numba import njit
from scipy.special import ive

__all__ = ["U1LatticeGauge", "u1_2d_wilson_loop_exact", "u1_2d_string_tension", "creutz_ratio"]


def _neighbor_tables(L: int, dim: int) -> tuple[np.ndarray, np.ndarray]:
    """fwd[mu, s], bwd[mu, s]: flat index of the site one step along +/- mu."""
    coords = np.array(np.unravel_index(np.arange(L**dim), (L,) * dim))
    fwd = np.empty((dim, L**dim), dtype=np.int64)
    bwd = np.empty((dim, L**dim), dtype=np.int64)
    for mu in range(dim):
        shift = np.zeros(dim, dtype=int)
        shift[mu] = 1
        fwd[mu] = np.ravel_multi_index(tuple((coords + shift[:, None]) % L), (L,) * dim)
        bwd[mu] = np.ravel_multi_index(tuple((coords - shift[:, None]) % L), (L,) * dim)
    return fwd, bwd


@njit
def _seed(seed):
    np.random.seed(seed)


@njit
def _plaquette_angle(theta, fwd, s, mu, nu):
    return theta[mu, s] + theta[nu, fwd[mu, s]] - theta[mu, fwd[nu, s]] - theta[nu, s]


@njit
def _local_action(theta, fwd, bwd, s, mu, beta):
    """-beta * sum of cos over the 2(d-1) plaquettes containing link (s, mu)."""
    dim = theta.shape[0]
    total = 0.0
    for nu in range(dim):
        if nu == mu:
            continue
        total += np.cos(_plaquette_angle(theta, fwd, s, mu, nu))
        total += np.cos(_plaquette_angle(theta, fwd, bwd[nu, s], mu, nu))
    return -beta * total


@njit
def _metropolis_sweeps(theta, fwd, bwd, beta, n_sweeps, step, n_hits):
    dim, V = theta.shape
    accepted = 0
    for _ in range(n_sweeps):
        for s in range(V):
            for mu in range(dim):
                for _h in range(n_hits):
                    old = theta[mu, s]
                    S_old = _local_action(theta, fwd, bwd, s, mu, beta)
                    theta[mu, s] = old + step * (2.0 * np.random.random() - 1.0)
                    dS = _local_action(theta, fwd, bwd, s, mu, beta) - S_old
                    if dS > 0.0 and np.random.random() >= np.exp(-dS):
                        theta[mu, s] = old
                    else:
                        accepted += 1
    return accepted / (n_sweeps * V * dim * n_hits)


@njit
def _mean_plaquette(theta, fwd):
    dim, V = theta.shape
    total = 0.0
    count = 0
    for s in range(V):
        for mu in range(dim):
            for nu in range(mu + 1, dim):
                total += np.cos(_plaquette_angle(theta, fwd, s, mu, nu))
                count += 1
    return total / count


@njit
def _mean_wilson_loop(theta, fwd, R, T):
    """Average of cos(loop angle) over all R x T rectangles, both orientations, all planes."""
    dim, V = theta.shape
    total = 0.0
    count = 0
    for mu in range(dim):
        for nu in range(dim):
            if mu == nu:
                continue
            for s0 in range(V):
                angle = 0.0
                s = s0
                for _ in range(R):  # bottom edge, +mu
                    angle += theta[mu, s]
                    s = fwd[mu, s]
                for _ in range(T):  # right edge, +nu
                    angle += theta[nu, s]
                    s = fwd[nu, s]
                s = fwd[nu, s0]
                for _ in range(T - 1):
                    s = fwd[nu, s]
                for _ in range(R):  # top edge traversed backwards
                    angle -= theta[mu, s]
                    s = fwd[mu, s]
                s = s0
                for _ in range(T):  # left edge traversed backwards
                    angle -= theta[nu, s]
                    s = fwd[nu, s]
                total += np.cos(angle)
                count += 1
    return total / count


class U1LatticeGauge:
    r"""Monte Carlo simulation of compact U(1) lattice gauge theory on a periodic :math:`L^d` lattice.

    Link angles are updated by Metropolis sweeps with the Wilson action
    (module docstring); the proposal width adapts during
    :meth:`thermalize` towards roughly 50% acceptance.

    Parameters
    ----------
    L : int
        Linear lattice size.
    beta : float
        Inverse coupling :math:`\beta = 1/g^2`.
    dim : int, default=2
        Space-time dimension (2 or 3).
    seed : int, optional
        Seed for numba's random generator (reproducible runs).
    hot_start : bool, default=True
        Random initial links; otherwise all links zero ("cold").

    Examples
    --------
    A cold (ordered) start has every plaquette equal to 1:

    >>> lat = U1LatticeGauge(L=4, beta=1.0, hot_start=False)
    >>> lat.plaquette()
    1.0
    """

    def __init__(self, L: int, beta: float, dim: int = 2, seed: int | None = None, hot_start: bool = True) -> None:
        if dim < 2:
            raise ValueError("a gauge theory needs dim >= 2")
        self.L, self.beta, self.dim = L, float(beta), dim
        self._fwd, self._bwd = _neighbor_tables(L, dim)
        rng = np.random.default_rng(seed)
        if seed is not None:
            _seed(seed)
        self.theta = rng.uniform(-np.pi, np.pi, (dim, L**dim)) if hot_start else np.zeros((dim, L**dim))
        self.step = 2.0
        self.acceptance = np.nan

    def sweep(self, n_sweeps: int = 1, n_hits: int = 2) -> float:
        """Run Metropolis sweeps (``n_hits`` trials per link per sweep); return the acceptance rate."""
        self.acceptance = _metropolis_sweeps(self.theta, self._fwd, self._bwd, self.beta, n_sweeps, self.step, n_hits)
        return self.acceptance

    def thermalize(self, n_sweeps: int = 200) -> None:
        """Equilibrate, tuning the proposal width towards ~50% acceptance."""
        for _ in range(max(1, n_sweeps // 10)):
            acc = self.sweep(10)
            self.step = float(np.clip(self.step * (0.5 + acc), 0.05, 2 * np.pi))

    def plaquette(self) -> float:
        r"""Current average plaquette :math:`\langle\cos\theta_P\rangle`."""
        return float(_mean_plaquette(self.theta, self._fwd))

    def wilson_loop(self, R: int, T: int) -> float:
        r"""Current :math:`R\times T` Wilson loop, averaged over positions, planes and orientations."""
        return float(_mean_wilson_loop(self.theta, self._fwd, R, T))

    def gauge_transform(self, alpha: np.ndarray) -> None:
        r"""Apply :math:`\theta_\mu(x) \to \theta_\mu(x) + \alpha(x) - \alpha(x + \hat\mu)`.

        Plaquettes and Wilson loops are invariant.

        Parameters
        ----------
        alpha : numpy.ndarray
            One angle per site, shape ``(L**dim,)``.
        """
        for mu in range(self.dim):
            self.theta[mu] += alpha - alpha[self._fwd[mu]]

    def measure(self, loops: list[tuple[int, int]], n_measurements: int = 200, sweeps_between: int = 5) -> dict:
        r"""Sample the plaquette and Wilson loops; return means and standard errors.

        Standard errors use 10 blocks of consecutive measurements
        (binning), which absorbs residual autocorrelation.

        Parameters
        ----------
        loops : list of (int, int)
            Loop sizes :math:`(R, T)`.
        n_measurements : int, default=200
            Number of measurements (a multiple of 10 is best).
        sweeps_between : int, default=5
            Sweeps between measurements.

        Returns
        -------
        dict
            ``{"plaquette": (mean, err), (R, T): (mean, err), ...}``.
        """
        keys: list[str | tuple[int, int]] = ["plaquette", *loops]
        data: dict[str | tuple[int, int], list[float]] = {key: [] for key in keys}
        for _ in range(n_measurements):
            self.sweep(sweeps_between)
            data["plaquette"].append(self.plaquette())
            for R, T in loops:
                data[(R, T)].append(self.wilson_loop(R, T))
        out = {}
        for key, vals in data.items():
            blocks = np.array_split(np.asarray(vals), 10)
            means = np.array([b.mean() for b in blocks])
            out[key] = (float(np.mean(vals)), float(means.std(ddof=1) / np.sqrt(len(means))))
        return out


def u1_2d_wilson_loop_exact(beta: float, area: np.ndarray, volume: int | None = None) -> np.ndarray:
    r"""Exact Wilson loop of 2D compact U(1) gauge theory with the Wilson action.

    In two dimensions the plaquette angles are independent variables (up
    to one global constraint on a torus), and a contractible loop is the
    product of the plaquettes it encloses. On an infinite lattice

    .. math::

        W(A) = \left(\frac{I_1(\beta)}{I_0(\beta)}\right)^{A},

    an exact area law, and on a periodic lattice of :math:`V` plaquettes

    .. math::

        W(A) = \frac{\sum_{n\in\mathbb Z} I_{n+1}(\beta)^{A}\,I_n(\beta)^{V-A}}
                    {\sum_{n\in\mathbb Z} I_n(\beta)^{V}}

    (Balian, Drouffe and Itzykson, Phys. Rev. D 11, 2098 (1975); Rusakov,
    Mod. Phys. Lett. A 5, 693 (1990)).

    Parameters
    ----------
    beta : float
        Inverse coupling.
    area : array_like
        Loop area :math:`A = RT` in plaquettes.
    volume : int, optional
        Number of plaquettes :math:`V = L^2` of a periodic lattice
        (``None``: infinite lattice).

    Returns
    -------
    numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> from scipy.special import iv
    >>> bool(np.isclose(u1_2d_wilson_loop_exact(2.0, 1), iv(1, 2.0) / iv(0, 2.0)))
    True
    """
    A = np.asarray(area, dtype=float)
    if volume is None:
        return (ive(1, beta) / ive(0, beta)) ** A
    n = np.arange(-60, 61)
    # exponentially scaled Bessel functions: common e^{beta V} factors cancel
    log_I = np.log(np.maximum(ive(n, beta), 1e-300))
    log_I1 = np.log(np.maximum(ive(n + 1, beta), 1e-300))
    A_ = np.atleast_1d(A)[:, None]
    log_num = A_ * log_I1 + (volume - A_) * log_I
    log_den = volume * log_I
    shift = log_den.max()
    W = np.exp(log_num - shift).sum(axis=1) / np.exp(log_den - shift).sum()
    return W if A.ndim else W[0]


def u1_2d_string_tension(beta: float) -> float:
    r"""Exact 2D U(1) string tension :math:`\sigma = -\ln[I_1(\beta)/I_0(\beta)]`, nonzero for every :math:`\beta`.

    Parameters
    ----------
    beta : float

    Returns
    -------
    float

    Examples
    --------
    >>> round(u1_2d_string_tension(1.0), 6)
    0.806562
    """
    return float(-np.log(ive(1, beta) / ive(0, beta)))


def creutz_ratio(W: dict, R: int, T: int) -> float:
    r"""Creutz ratio :math:`\chi(R,T) = -\ln\frac{W(R,T)\,W(R-1,T-1)}{W(R,T-1)\,W(R-1,T)}`.

    Perimeter and constant terms cancel, leaving the string tension
    :math:`\sigma` for an area law (Creutz 1980).

    Parameters
    ----------
    W : dict
        Maps ``(R, T)`` to the Wilson loop value (or ``(value, error)``).
        A missing ``(R, T)`` falls back to ``(T, R)`` (equal for loops
        averaged over orientations, as :meth:`U1LatticeGauge.measure`
        returns), and a ``(0, T)`` or ``(R, 0)`` loop counts as 1.
    R, T : int

    Returns
    -------
    float

    Examples
    --------
    >>> import numpy as np
    >>> W = {(r, t): np.exp(-0.3 * r * t - 0.1 * (r + t)) for r in range(1, 4) for t in range(1, 4)}
    >>> round(creutz_ratio(W, 2, 2), 12)
    0.3
    """

    def w(r, t):
        if r == 0 or t == 0:
            return 1.0
        val = W[(r, t)] if (r, t) in W else W[(t, r)]
        return val[0] if isinstance(val, tuple) else val

    return float(-np.log(w(R, T) * w(R - 1, T - 1) / (w(R, T - 1) * w(R - 1, T))))
