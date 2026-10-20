"""
Cross-check of physicskit's shared adaptive integrator against SciPy.

physicskit's ``dopri5`` is a numba-compiled Dormand-Prince RK5(4), the
tableau of SciPy's ``RK45``. Two comparisons:

1. Fixed steps: ``dopri5_step`` repeated on the Lorenz system, against
   SciPy's ``RK45`` held to the same constant step (tolerances so loose
   that every step is accepted). The two must agree to round-off, step by
   step, before chaos amplifies it.
2. Adaptive steps: ``dopri5_integrate`` on an eccentric Kepler orbit over
   one period, against SciPy's eighth-order ``DOP853`` at a much tighter
   tolerance, and against the exact return to the initial state.

Each comparison is asserted; ``RESULTS`` holds the largest deviations.
"""

import numpy as np
from numba import njit
from scipy.integrate import RK45, solve_ivp

from physicskit.integrators import dopri5_integrate, dopri5_step

RESULTS = {}


def record(name, dev, tol):
    RESULTS[name] = float(dev)
    assert dev < tol, (name, dev)
    print(f"{name:60s} max deviation {dev:.1e}")


@njit
def lorenz(s, t, p):
    return np.array([p[0] * (s[1] - s[0]), s[0] * (p[1] - s[2]) - s[1], s[0] * s[1] - p[2] * s[2]])


@njit
def kepler(s, t, p):
    r3 = (s[0] ** 2 + s[1] ** 2) ** 1.5
    return np.array([s[2], s[3], -s[0] / r3, -s[1] / r3])


# 1. Fixed steps on the Lorenz system ---------------------------------------

params = np.array([10.0, 28.0, 8.0 / 3.0])
s0, h, n = np.array([1.0, 1.0, 1.0]), 1e-3, 2000  # to t = 2, before trajectories diverge
ref = RK45(lambda t, s: lorenz(s, t, params), 0.0, s0, t_bound=n * h, first_step=h, max_step=h, rtol=1e3, atol=1e3)
s, dev = s0.copy(), 0.0
for i in range(n):
    s, _ = dopri5_step(lorenz, s, i * h, h, params)
    ref.step()
    dev = max(dev, np.abs(s - ref.y).max() / np.abs(ref.y).max())
record("dopri5 vs RK45: 2000 fixed Dormand-Prince steps (Lorenz)", dev, 1e-12)

# 2. Adaptive steps on a Kepler orbit ---------------------------------------

e = 0.6
s0 = np.array([1.0 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])  # perihelion, a = 1, GM = 1
T = 2 * np.pi
_, states = dopri5_integrate(kepler, s0, 0.0, T, 1e-3, np.zeros(0), rtol=1e-12, atol=1e-12)
exact = solve_ivp(lambda t, s: kepler(s, t, None), (0.0, T), s0, method="DOP853", rtol=1e-13, atol=1e-14).y[:, -1]
record("dopri5 (rtol 1e-12) vs DOP853: Kepler orbit, e = 0.6", np.abs(states[-1] - exact).max(), 1e-9)
record("dopri5 (rtol 1e-12): return to perihelion after one period", np.abs(states[-1] - s0).max(), 1e-9)
