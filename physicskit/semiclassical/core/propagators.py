r"""Semiclassical propagators: the Van Vleck-Morette determinant and Herman-Kluk frozen Gaussians.

Both propagators here reconstruct the quantum time-evolution operator
from *classical* trajectories alone. Van Vleck-Morette theory
(:func:`van_vleck_propagator_1d`) uses a single trajectory connecting a
fixed start and end point, dressed with a stability prefactor built from
its **monodromy matrix** -- the linearized map
:math:`(\delta q_0,\delta p_0)\mapsto(\delta q_t,\delta p_t)` obtained by
propagating small deviations alongside the trajectory
(:func:`propagate_trajectory_monodromy_action`). Herman and Kluk's 1984
frozen-Gaussian method (:func:`herman_kluk_propagate_wavepacket`) instead
sums the contributions of *many* such trajectories, one launched from
each point of a phase-space grid under the initial wavepacket, each one
carrying its own rigid (frozen-width) Gaussian, monodromy-built
prefactor, and classical action phase.

Right-hand-side functions passed to the trajectory/monodromy integrator
must be module-level ``@njit`` functions with signature
``dVdx(q, params) -> float`` / ``d2Vdx2(q, params) -> float`` /
``V(q, params) -> float``, with ``params`` a ``float64`` array -- the
same convention used throughout this package's Numba-accelerated
integrators, letting a single compiled kernel serve any potential.
"""

from __future__ import annotations

import numpy as np
from numba import njit

__all__ = [
    "propagate_trajectory_monodromy_action",
    "van_vleck_prefactor",
    "count_caustics",
    "van_vleck_propagator_1d",
    "frozen_gaussian_1d",
    "coherent_state_overlap",
    "herman_kluk_prefactor",
    "herman_kluk_propagate_wavepacket",
]


@njit(cache=True)
def _rhs(state, dVdx, d2Vdx2, V, m, params):
    q, p, M00, M01, M10, M11, _S = state
    d2V = d2Vdx2(q, params)
    out = np.empty(7)
    out[0] = p / m
    out[1] = -dVdx(q, params)
    out[2] = M10 / m
    out[3] = M11 / m
    out[4] = -d2V * M00
    out[5] = -d2V * M01
    out[6] = p * p / (2.0 * m) - V(q, params)
    return out


@njit(cache=True)
def _rk4_step(state, dt, dVdx, d2Vdx2, V, m, params):
    k1 = _rhs(state, dVdx, d2Vdx2, V, m, params)
    k2 = _rhs(state + 0.5 * dt * k1, dVdx, d2Vdx2, V, m, params)
    k3 = _rhs(state + 0.5 * dt * k2, dVdx, d2Vdx2, V, m, params)
    k4 = _rhs(state + dt * k3, dVdx, d2Vdx2, V, m, params)
    return state + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


@njit(cache=True)
def _integrate(q0, p0, dt, steps, dVdx, d2Vdx2, V, m, params):
    state = np.array([q0, p0, 1.0, 0.0, 0.0, 1.0, 0.0])
    Mqp_hist = np.empty(steps + 1)
    Mqp_hist[0] = 0.0
    for i in range(steps):
        state = _rk4_step(state, dt, dVdx, d2Vdx2, V, m, params)
        Mqp_hist[i + 1] = state[3]
    return state, Mqp_hist


def propagate_trajectory_monodromy_action(q0: float, p0: float, dVdx, d2Vdx2, V, m: float, dt: float, steps: int, params: np.ndarray | None = None):
    r"""Integrate a classical trajectory together with its monodromy matrix and Hamilton principal function.

    Advances the Hamilton equations :math:`\dot q=p/m,\ \dot p=-V'(q)`
    together with the variational (tangent) equations for small
    deviations, :math:`\dot{\delta q}=\delta p/m,\ \dot{\delta p}
    =-V''(q)\delta q`, whose solution operator is the monodromy matrix
    :math:`M(t)=\begin{pmatrix}\partial q_t/\partial q_0 &
    \partial q_t/\partial p_0\\ \partial p_t/\partial q_0 &
    \partial p_t/\partial p_0\end{pmatrix}`, and accumulates the
    classical action (Hamilton's principal function)
    :math:`S(t)=\int_0^t\left[\tfrac{p^2}{2m}-V(q)\right]dt'` along the
    way -- everything :func:`van_vleck_propagator_1d` and
    :func:`herman_kluk_propagate_wavepacket` need from one trajectory,
    computed in a single RK4 pass.

    Because the flow is Hamiltonian, :math:`\det M(t)=1` exactly for all
    :math:`t` (Liouville's theorem); this is a good numerical sanity
    check on any trajectory this function returns.

    Parameters
    ----------
    q0, p0 : float
        Initial position and momentum.
    dVdx : callable
        Numba-jitted force law ``dVdx(q, params) -> float``, :math:`V'(q)`.
    d2Vdx2 : callable
        Numba-jitted curvature ``d2Vdx2(q, params) -> float``, :math:`V''(q)`.
    V : callable
        Numba-jitted potential ``V(q, params) -> float``.
    m : float
        Particle mass.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps (total propagation time is ``dt * steps``).
    params : ndarray, optional
        Parameter vector passed through to ``dVdx``/``d2Vdx2``/``V``.
        Defaults to an empty array.

    Returns
    -------
    q_t, p_t : float
        Final position and momentum.
    M : ndarray, shape (2, 2)
        Monodromy matrix at time ``t = dt * steps``.
    S : float
        Classical action accumulated along the trajectory.
    Mqp_history : ndarray, shape (steps + 1,)
        :math:`\partial q_t/\partial p_0` at every step, for
        :func:`count_caustics`.

    See Also
    --------
    van_vleck_prefactor : Turns ``M`` into a propagator amplitude.
    count_caustics : Turns ``Mqp_history`` into a Maslov index.

    Examples
    --------
    For the harmonic oscillator, the monodromy matrix has the exact
    closed form :math:`\begin{pmatrix}\cos\omega t & \sin(\omega
    t)/(m\omega)\\ -m\omega\sin\omega t & \cos\omega t\end{pmatrix}`:

    >>> import numpy as np
    >>> from numba import njit
    >>> m, omega = 1.0, 1.0
    >>> params = np.array([m * omega ** 2])
    >>> dVdx = njit(lambda q, params: params[0] * q, cache=False)
    >>> d2Vdx2 = njit(lambda q, params: params[0], cache=False)
    >>> V = njit(lambda q, params: 0.5 * params[0] * q ** 2, cache=False)
    >>> t = 1.3
    >>> q_t, p_t, M, S, _ = propagate_trajectory_monodromy_action(1.0, 0.3, dVdx, d2Vdx2, V, m, dt=t / 4000, steps=4000, params=params)
    >>> M_exact = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
    >>> bool(np.max(np.abs(M - M_exact)) < 1e-6)
    True
    >>> round(float(np.linalg.det(M)), 8)
    1.0
    """
    if params is None:
        params = np.empty(0)
    state, Mqp_hist = _integrate(float(q0), float(p0), float(dt), int(steps), dVdx, d2Vdx2, V, float(m), params)
    q_t, p_t = state[0], state[1]
    M = np.array([[state[2], state[3]], [state[4], state[5]]])
    S = state[6]
    return q_t, p_t, M, S, Mqp_hist


def van_vleck_prefactor(Mqp: float, hbar: float = 1.0) -> float:
    r"""Van Vleck-Morette amplitude :math:`\sqrt{1/(2\pi\hbar\,|\partial q_t/\partial p_0|)}`.

    The prefactor is the square root of (minus) the mixed second
    derivative of the classical action,
    :math:`\left|\partial^2S/\partial q_0\partial q_t\right| =
    1/|\partial q_t/\partial p_0|` -- large where nearby trajectories
    stay close together (many classical paths reinforce the same
    endpoint) and formally divergent at a focal point
    (:math:`\partial q_t/\partial p_0=0`, where infinitesimally
    different initial momenta all reconverge on the same final position).

    Parameters
    ----------
    Mqp : float
        The :math:`(0,1)` element of the monodromy matrix,
        :math:`\partial q_t/\partial p_0`, from
        :func:`propagate_trajectory_monodromy_action`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    float
        The (real, non-negative) prefactor amplitude.

    Examples
    --------
    >>> round(van_vleck_prefactor(Mqp=1.0, hbar=1.0), 6)
    0.398942
    """
    return float(np.sqrt(1.0 / (2.0 * np.pi * hbar * abs(Mqp))))


def count_caustics(Mqp_history: np.ndarray) -> int:
    r"""Count sign changes of :math:`\partial q_t/\partial p_0` along a trajectory -- the Maslov index.

    Each time :math:`\partial q_t/\partial p_0` passes through zero, the
    trajectory crosses a focal point (conjugate point / caustic), and the
    semiclassical propagator picks up an extra phase of :math:`-\pi/2`
    (Gutzwiller 1967; see also the :math:`-\pi/4` phase at a single WKB
    turning point in :func:`physicskit.semiclassical.core.wkb.wkb_wavefunction`,
    which is this same phenomenon in the time-independent picture).

    Parameters
    ----------
    Mqp_history : ndarray
        :math:`\partial q_t/\partial p_0` sampled along the trajectory,
        from :func:`propagate_trajectory_monodromy_action`; the first
        entry (always 0, at :math:`t=0`) is ignored.

    Returns
    -------
    int
        The Maslov index :math:`\mu`.

    See Also
    --------
    van_vleck_propagator_1d : Uses this to fix the propagator's overall phase.

    Examples
    --------
    >>> count_caustics(np.array([0.0, 1.0, 0.5, -0.3, -0.8, 0.2]))
    2
    """
    nonzero = Mqp_history[1:]
    nonzero = nonzero[nonzero != 0]
    return int(np.sum(np.diff(np.sign(nonzero)) != 0))


def van_vleck_propagator_1d(q0: float, p0: float, dVdx, d2Vdx2, V, m: float, dt: float, steps: int, hbar: float = 1.0, params: np.ndarray | None = None):
    r"""Semiclassical (Van Vleck-Morette) propagator amplitude along one classical trajectory.

    .. math::

       K(q_t,t;q_0,0) \approx \sqrt{\frac{1}{2\pi\hbar\,|\partial q_t/\partial p_0|}}\,
       \exp\!\left[\frac{i}{\hbar}S(t) - i\frac{\pi}{4} - i\mu\frac{\pi}{2}\right],

    the leading-order (:math:`\hbar\to0`) approximation to the quantum
    propagator, exact whenever the potential is at most quadratic (the
    free particle and the harmonic oscillator), since then the WKB
    expansion this formula comes from truncates exactly.

    Parameters
    ----------
    q0, p0 : float
        Initial position and momentum. ``p0`` fixes which trajectory
        (and hence which final position ``q_t``) is used; this function
        does not solve the two-point boundary-value problem of finding
        the ``p0`` that reaches a *prescribed* ``q_t``.
    dVdx, d2Vdx2, V : callable
        Numba-jitted potential derivatives and the potential itself, as
        in :func:`propagate_trajectory_monodromy_action`.
    m : float
        Particle mass.
    dt : float
        Time step.
    steps : int
        Number of RK4 steps.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    params : ndarray, optional
        Parameter vector passed through to ``dVdx``/``d2Vdx2``/``V``.

    Returns
    -------
    q_t : float
        The trajectory's final position.
    K : complex
        The semiclassical propagator amplitude :math:`K(q_t,t;q_0,0)`.

    See Also
    --------
    propagate_trajectory_monodromy_action : Supplies the trajectory, monodromy, and action.
    herman_kluk_propagate_wavepacket : Sums many such trajectories into a full wavepacket propagator.

    Examples
    --------
    A free particle's semiclassical propagator matches the exact quantum
    propagator :math:`\sqrt{m/(2\pi i\hbar t)}\exp[im(q_t-q_0)^2/(2\hbar t)]` exactly:

    >>> import numpy as np
    >>> from numba import njit
    >>> zero = njit(lambda q, params: 0.0, cache=False)
    >>> q_t, K = van_vleck_propagator_1d(q0=0.0, p0=1.0, dVdx=zero, d2Vdx2=zero, V=zero, m=1.0, dt=1.0 / 2000, steps=2000)
    >>> K_exact = np.sqrt(1.0 / (2j * np.pi * 1.0)) * np.exp(1j * (q_t - 0.0) ** 2 / 2.0)
    >>> bool(abs(K - K_exact) < 1e-6)
    True

    A harmonic-oscillator trajectory that has not yet crossed a focal
    point (:math:`t<\pi/\omega`) matches the exact Van Vleck (Mehler)
    propagator just as closely:

    >>> m, omega = 1.0, 1.0
    >>> params = np.array([m * omega ** 2])
    >>> dVdx = njit(lambda q, params: params[0] * q, cache=False)
    >>> d2Vdx2 = njit(lambda q, params: params[0], cache=False)
    >>> V = njit(lambda q, params: 0.5 * params[0] * q ** 2, cache=False)
    >>> q0, p0, t = 1.0, 0.3, 1.3
    >>> q_t, K = van_vleck_propagator_1d(q0, p0, dVdx, d2Vdx2, V, m, dt=t / 4000, steps=4000, params=params)
    >>> K_exact = (np.sqrt(m * omega / (2j * np.pi * np.sin(omega * t)))
    ...            * np.exp(1j * m * omega / (2 * np.sin(omega * t)) * ((q_t ** 2 + q0 ** 2) * np.cos(omega * t) - 2 * q_t * q0)))
    >>> bool(abs(K - K_exact) < 1e-5)
    True
    """
    q_t, _p_t, M, S, Mqp_hist = propagate_trajectory_monodromy_action(q0, p0, dVdx, d2Vdx2, V, m, dt, steps, params)
    mu = count_caustics(Mqp_hist)
    prefactor = van_vleck_prefactor(M[0, 1], hbar)
    phase = S / hbar - np.pi / 4.0 - mu * np.pi / 2.0
    K = prefactor * np.exp(1j * phase)
    return q_t, K


def frozen_gaussian_1d(x: np.ndarray, qc: float, pc: float, gamma: float, hbar: float = 1.0) -> np.ndarray:
    r"""A normalized, fixed-width ("frozen") Gaussian wavepacket (coherent state).

    .. math::

       g_{q_c,p_c}(x) = \left(\frac{2\gamma}{\pi}\right)^{1/4}
       \exp\!\left[-\gamma(x-q_c)^2 + \frac{i}{\hbar}p_c(x-q_c)\right].

    The building block of the Herman-Kluk method: every trajectory
    carries one of these, always at the fixed width set by :math:`\gamma`
    (hence "frozen"), riding on top of its classical phase-space point
    :math:`(q_c,p_c)`.

    Parameters
    ----------
    x : ndarray
        Positions at which to evaluate the wavepacket.
    qc, pc : float
        Center of the Gaussian in position and momentum.
    gamma : float
        Width parameter (inverse squared length); larger :math:`\gamma`
        means a narrower, more position-localized packet.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    ndarray of complex
        :math:`g_{q_c,p_c}(x)`, same shape as ``x``.

    See Also
    --------
    coherent_state_overlap : The closed-form overlap of two such Gaussians.

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(-20, 20, 4000)
    >>> psi = frozen_gaussian_1d(x, qc=1.0, pc=2.0, gamma=0.5)
    >>> round(float(np.trapezoid(np.abs(psi) ** 2, x)), 8)
    1.0
    """
    norm = (2.0 * gamma / np.pi) ** 0.25
    return norm * np.exp(-gamma * (x - qc) ** 2 + 1j * pc * (x - qc) / hbar)


def coherent_state_overlap(q1: float, p1: float, q2: float, p2: float, gamma: float, hbar: float = 1.0) -> complex:
    r"""Closed-form overlap :math:`\langle g_{q_1,p_1}|g_{q_2,p_2}\rangle` of two equal-width frozen Gaussians.

    .. math::

       \langle g_1|g_2\rangle = \exp\!\left[-\frac{\gamma}{2}(q_1-q_2)^2
       - \frac{(p_1-p_2)^2}{8\gamma\hbar^2}
       + \frac{i}{2\hbar}(q_1-q_2)(p_1+p_2)\right].

    Parameters
    ----------
    q1, p1 : float
        Phase-space center of the bra state.
    q2, p2 : float
        Phase-space center of the ket state.
    gamma : float
        Shared width parameter, as in :func:`frozen_gaussian_1d`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    complex
        The overlap :math:`\langle g_1|g_2\rangle`.

    See Also
    --------
    frozen_gaussian_1d : The states being overlapped.

    Examples
    --------
    Matches direct numerical integration of the two wavepackets:

    >>> import numpy as np
    >>> x = np.linspace(-40, 40, 20000)
    >>> gamma = 1.0
    >>> psi1 = frozen_gaussian_1d(x, qc=0.3, pc=0.7, gamma=gamma)
    >>> psi2 = frozen_gaussian_1d(x, qc=-0.2, pc=1.1, gamma=gamma)
    >>> numeric = np.trapezoid(np.conj(psi1) * psi2, x)
    >>> closed_form = coherent_state_overlap(0.3, 0.7, -0.2, 1.1, gamma)
    >>> bool(abs(numeric - closed_form) < 1e-6)
    True

    A state's overlap with itself is 1:

    >>> round(float(abs(coherent_state_overlap(1.0, 2.0, 1.0, 2.0, gamma=0.8))), 8)
    1.0
    """
    return np.exp(-gamma / 2.0 * (q1 - q2) ** 2 - (p1 - p2) ** 2 / (8.0 * gamma * hbar**2) + 1j * (q1 - q2) * (p1 + p2) / (2.0 * hbar))


def herman_kluk_prefactor(M: np.ndarray, gamma: float, hbar: float = 1.0) -> complex:
    r"""Herman-Kluk prefactor :math:`C_t(q_0,p_0)` built from the monodromy matrix.

    .. math::

       C_t = \sqrt{\frac{1}{2}\left(M_{qq} + M_{pp}
       - i\hbar\gamma M_{qp} + \frac{i}{\hbar\gamma}M_{pq}\right)},

    which reduces to :math:`C_t=1` at :math:`t=0` (:math:`M=\mathbb{1}`),
    consistent with the frozen Gaussian basis's resolution of the
    identity, :math:`\int \tfrac{dq\,dp}{2\pi\hbar}\,|g_{q,p}\rangle\langle
    g_{q,p}| = \hat{1}`.

    Parameters
    ----------
    M : ndarray, shape (2, 2)
        Monodromy matrix from :func:`propagate_trajectory_monodromy_action`.
    gamma : float
        Frozen-Gaussian width parameter, as in :func:`frozen_gaussian_1d`.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.

    Returns
    -------
    complex
        The prefactor :math:`C_t`.

    Examples
    --------
    >>> import numpy as np
    >>> complex(herman_kluk_prefactor(np.eye(2), gamma=1.0))
    (1+0j)
    """
    Mqq, Mqp, Mpq, Mpp = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    val = 0.5 * (Mqq + Mpp - 1j * hbar * gamma * Mqp + 1j * Mpq / (hbar * gamma))
    return np.sqrt(val + 0j)


def herman_kluk_propagate_wavepacket(
    qc0: float,
    pc0: float,
    gamma: float,
    dVdx,
    d2Vdx2,
    V,
    m: float,
    dt: float,
    steps: int,
    x_eval: np.ndarray,
    hbar: float = 1.0,
    n_grid: int = 41,
    n_sigma: float = 6.0,
    params: np.ndarray | None = None,
) -> np.ndarray:
    r"""Propagate an initial frozen-Gaussian wavepacket with the multi-trajectory Herman-Kluk method.

    Launches one classical trajectory from every point of a regular grid
    covering the initial coherent state's phase-space support, and sums
    their frozen-Gaussian contributions

    .. math::

       \psi(x,t) \approx \int\!\frac{dq_0\,dp_0}{2\pi\hbar}\,
       C_t(q_0,p_0)\,e^{iS(t)/\hbar}\,
       \langle g_{q_0,p_0}|\psi_0\rangle\, g_{q_t,p_t}(x),

    approximating the integral over initial conditions by a Riemann sum
    on a grid spanning :math:`\pm n_\sigma` standard deviations of the
    initial coherent state's own phase-space Gaussian
    (:math:`\sigma_q=1/(2\sqrt\gamma)`, :math:`\sigma_p=\hbar\sqrt\gamma`).
    The per-trajectory classical propagation
    (:func:`propagate_trajectory_monodromy_action`) is Numba-compiled,
    since it is called once per grid point -- the dominant cost for any
    reasonably fine grid.

    In the :math:`t\to0` limit this reduces to the frozen-Gaussian
    resolution of the identity and reconstructs the initial wavepacket
    essentially exactly; away from that limit, this quadrature-grid
    evaluation of the Herman-Kluk integral does not exactly conserve
    :math:`\int|\psi|^2\,dx` (the true continuous phase-space integral
    does, for at-most-quadratic potentials) -- convergence in ``n_grid``,
    ``n_sigma``, and ``gamma`` should be checked for any serious use.

    Parameters
    ----------
    qc0, pc0 : float
        Center of the initial frozen-Gaussian wavepacket.
    gamma : float
        Width parameter shared by the initial state and every frozen
        Gaussian in the propagation.
    dVdx, d2Vdx2, V : callable
        Numba-jitted potential derivatives and the potential itself, as
        in :func:`propagate_trajectory_monodromy_action`.
    m : float
        Particle mass.
    dt : float
        Time step for each trajectory's RK4 integration.
    steps : int
        Number of RK4 steps (propagation time is ``dt * steps``).
    x_eval : ndarray
        Positions at which to evaluate the propagated wavepacket.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    n_grid : int, default=41
        Number of grid points along each of the :math:`q_0`, :math:`p_0` axes.
    n_sigma : float, default=6.0
        Half-width of the sampling grid, in standard deviations of the
        initial state's phase-space Gaussian.
    params : ndarray, optional
        Parameter vector passed through to ``dVdx``/``d2Vdx2``/``V``.

    Returns
    -------
    ndarray of complex, shape matching ``x_eval``
        The propagated wavepacket :math:`\psi(x,t)`.

    See Also
    --------
    van_vleck_propagator_1d : The single-trajectory propagator this sums many copies of.
    frozen_gaussian_1d : The initial state and the basis each trajectory carries.

    Examples
    --------
    At very short times the sum reduces to the frozen-Gaussian
    resolution of the identity and reproduces the (unpropagated) initial
    coherent state to high accuracy:

    >>> import numpy as np
    >>> from numba import njit
    >>> m, omega = 1.0, 1.0
    >>> params = np.array([m * omega ** 2])
    >>> dVdx = njit(lambda q, params: params[0] * q, cache=False)
    >>> d2Vdx2 = njit(lambda q, params: params[0], cache=False)
    >>> V = njit(lambda q, params: 0.5 * params[0] * q ** 2, cache=False)
    >>> x = np.linspace(-6, 6, 400)
    >>> psi = herman_kluk_propagate_wavepacket(
    ...     1.0, 0.0, gamma=1.0, dVdx=dVdx, d2Vdx2=d2Vdx2, V=V, m=m, dt=1e-6, steps=1, x_eval=x, n_grid=61, n_sigma=7.0, params=params
    ... )
    >>> psi0 = frozen_gaussian_1d(x, qc=1.0, pc=0.0, gamma=1.0)
    >>> fidelity = abs(np.trapezoid(np.conj(psi0) * psi, x)) ** 2
    >>> bool(fidelity > 0.999)
    True
    """
    if params is None:
        params = np.empty(0)
    sigma_q = 1.0 / (2.0 * np.sqrt(gamma))
    sigma_p = hbar * np.sqrt(gamma)
    q0_grid = np.linspace(qc0 - n_sigma * sigma_q, qc0 + n_sigma * sigma_q, n_grid)
    p0_grid = np.linspace(pc0 - n_sigma * sigma_p, pc0 + n_sigma * sigma_p, n_grid)
    dq0 = q0_grid[1] - q0_grid[0]
    dp0 = p0_grid[1] - p0_grid[0]
    measure = dq0 * dp0 / (2.0 * np.pi * hbar)

    psi = np.zeros_like(x_eval, dtype=complex)
    for q0 in q0_grid:
        for p0 in p0_grid:
            q_t, p_t, M, S, _ = propagate_trajectory_monodromy_action(q0, p0, dVdx, d2Vdx2, V, m, dt, steps, params)
            C = herman_kluk_prefactor(M, gamma, hbar)
            ov = coherent_state_overlap(q0, p0, qc0, pc0, gamma, hbar)
            psi += measure * C * np.exp(1j * S / hbar) * ov * frozen_gaussian_1d(x_eval, q_t, p_t, gamma, hbar)
    return psi
