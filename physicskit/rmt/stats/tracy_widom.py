"""The Tracy-Widom distributions F_1, F_2, F_4: limiting law of the
largest eigenvalue at the soft edge of the Gaussian ensembles.

References
----------
C. A. Tracy, H. Widom, "Level-spacing distributions and the Airy
kernel", Commun. Math. Phys. 159 (1994) 151 -- F_2 (GUE, beta=2).
C. A. Tracy, H. Widom, "On orthogonal and symplectic matrix ensembles",
Commun. Math. Phys. 177 (1996) 727 -- F_1 (GOE, beta=1) and F_4
(GSE, beta=4).

Construction
------------
All three distributions are built from q(x), the Hastings-McLeod
solution of the Painleve II equation

    q''(x) = x*q(x) + 2*q(x)^3,   q(x) ~ Ai(x) as x -> +infinity

(the unique solution with this decaying tail; other solutions of the
same ODE blow up). q is obtained numerically by integrating *backward*
from a large positive x0 (where q(x0) = Ai(x0), q'(x0) = Ai'(x0) to
machine precision) down to a negative x_min -- forward integration from
-infinity is unstable, since nearby non-Hastings-McLeod solutions
diverge exponentially in that direction.

Numerical domain note: backward integration itself becomes unstable
(round-off eventually kicks the trajectory onto a diverging branch) for
x below about -9 to -10. This implementation restricts to [-6, 6] by
default, verified to match the known asymptotic q(x) ~ sqrt(-x/2) to
better than 0.2% at x=-6, and F_2/F_1/F_4 are already extremely close to
0 there (the left tail decays like ``exp(-|s|^3/12)``), so truncating the
domain has negligible effect on the distributions actually used.

Given q, the three distributions are:

    F_2(s) = exp(-integral_s^inf (x-s) q(x)^2 dx)
    F_1(s) = sqrt(F_2(s)) * exp(-(1/2) * integral_s^inf q(x) dx)
    F_4(s) = sqrt(F_2(s)) * cosh((1/2) * integral_s^inf q(x) dx)

Edge-scaling note (important, and the one detail here that is NOT
standard textbook boilerplate): for a Gaussian beta-ensemble normalized
to semicircle support [-2, 2] (as ``physicskit.rmt.ensembles.gaussian`` uses),
the largest eigenvalue converges as

    n^(2/3) * (lambda_max - 2)  ->  F_1   (GOE, beta=1)
    n^(2/3) * (lambda_max - 2)  ->  F_2   (GUE, beta=2)
    (2n)^(2/3) * (lambda_max - 2)  ->  F_4   (GSE, beta=4)

The GSE case's extra factor of 2 in the effective scale is easy to get
wrong from memory (and was NOT assumed here) -- it was determined by
directly comparing candidate scalings against Monte Carlo largest-
eigenvalue samples from the already-validated ``GSE`` ensemble, via a
KS test, rather than taken from a half-remembered formula. See the
design/development notes and ``tests/test_tracy_widom.py``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid, solve_ivp
from scipy.special import airy

from ..spectrum import Spectrum


@dataclass
class _PainleveIISolution:
    grid: np.ndarray
    F1: np.ndarray
    F2: np.ndarray
    F4: np.ndarray


_CACHE: dict[tuple[float, float, int], _PainleveIISolution] = {}


def _solve(x_max: float = 6.0, x_min: float = -6.0, n_grid: int = 20000) -> _PainleveIISolution:
    """Solve Painleve II once and build F1/F2/F4 on a grid. Cached by
    (x_max, x_min, n_grid) since this is a one-time cost reused by every
    subsequent CDF/PDF/sampling call."""
    key = (x_max, x_min, n_grid)
    if key in _CACHE:
        return _CACHE[key]

    def rhs(x: float, y: tuple[float, float]) -> list[float]:
        q, qp = y
        return [qp, x * q + 2.0 * q**3]

    ai, aip, _, _ = airy(x_max)
    sol = solve_ivp(
        rhs,
        [x_max, x_min],
        [ai, aip],
        dense_output=True,
        rtol=1e-12,
        atol=1e-14,
        max_step=0.005,
    )
    if not sol.success:
        raise RuntimeError(f"Painleve II integration failed: {sol.message}")

    grid = np.linspace(x_min, x_max, n_grid)
    q_vals = sol.sol(grid)[0]

    # Reverse cumulative integrals (needed since every quantity here is
    # defined as an integral from s to +infinity, i.e. against the grid's
    # natural ascending order).
    xs_rev = grid[::-1]
    q2_rev = (q_vals**2)[::-1]
    q_rev = q_vals[::-1]

    a_s = -cumulative_trapezoid(q2_rev, xs_rev, initial=0.0)[::-1]  # int_s^inf q^2 dx
    b_s = -cumulative_trapezoid(xs_rev * q2_rev, xs_rev, initial=0.0)[::-1]  # int_s^inf x q^2 dx
    i_s = -cumulative_trapezoid(q_rev, xs_rev, initial=0.0)[::-1]  # int_s^inf q dx

    r_s = b_s - grid * a_s
    f2 = np.exp(-r_s)
    f2_clipped = np.clip(f2, 0.0, 1.0)
    f1 = np.sqrt(f2_clipped) * np.exp(-0.5 * i_s)
    f4 = np.sqrt(f2_clipped) * np.cosh(0.5 * i_s)

    result = _PainleveIISolution(grid=grid, F1=f1, F2=np.clip(f2, 0.0, 1.0), F4=np.clip(f4, 0.0, 1.0))
    _CACHE[key] = result
    return result


# beta -> which precomputed curve, and the edge-scaling exponent base
# (n for beta=1,2; 2*n for beta=4 -- see module docstring).
_BETA_CURVE = {1: "F1", 2: "F2", 4: "F4"}
_BETA_SCALE_MULTIPLIER = {1: 1.0, 2: 1.0, 4: 2.0}


def largest_eigenvalues(spectrum: Spectrum) -> np.ndarray:
    """The largest (rescaled) eigenvalue from each sample in a Spectrum
    -- the raw statistic that Tracy-Widom-type soft-edge laws govern."""
    return spectrum.rescaled.max(axis=1)


def tracy_widom_cdf(s: np.ndarray, beta: int) -> np.ndarray:
    """CDF of the Tracy-Widom distribution at Dyson index beta (1, 2, or 4)."""
    if beta not in _BETA_CURVE:
        raise ValueError(
            f"Only beta in {{1, 2, 4}} have closed-form Tracy-Widom laws "
            f"(F_1, F_2, F_4); beta={beta} is not supported. The general-beta "
            f"soft-edge law exists (Ramirez-Rider-Virag stochastic Airy "
            f"operator) but has no elementary closed form."
        )
    sol = _solve()
    curve = getattr(sol, _BETA_CURVE[beta])
    s = np.asarray(s, dtype=float)
    return np.interp(s, sol.grid, curve, left=0.0, right=1.0)


def tracy_widom_edge_scale(n: int, beta: int) -> float:
    """Factor by which (lambda_max - 2) should be multiplied to compare
    against the beta-indexed Tracy-Widom CDF: n^(2/3) for beta=1, 2;
    (2n)^(2/3) for beta=4 (see module docstring for how the beta=4 factor
    was determined)."""
    if beta not in _BETA_SCALE_MULTIPLIER:
        raise ValueError(f"Only beta in {{1, 2, 4}} are supported, got {beta}")
    return (_BETA_SCALE_MULTIPLIER[beta] * n) ** (2.0 / 3.0)


def tracy_widom_rvs(size: int | tuple[int, ...], beta: int, rng: np.random.Generator) -> np.ndarray:
    """Sample from the Tracy-Widom distribution via inverse-CDF
    interpolation on the same grid used for the CDF."""
    if beta not in _BETA_CURVE:
        raise ValueError(f"Only beta in {{1, 2, 4}} are supported, got {beta}")
    sol = _solve()
    curve = getattr(sol, _BETA_CURVE[beta])
    # curve is monotonically increasing in practice; interpolate its inverse
    u = rng.uniform(curve[0], curve[-1], size=size)
    return np.interp(u, curve, sol.grid)
