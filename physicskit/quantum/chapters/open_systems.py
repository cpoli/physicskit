r"""Open quantum systems: the Lindblad master equation, quantum trajectories, and noise channels.

A system coupled to a memoryless (Markovian) environment evolves under the
Gorini-Kossakowski-Sudarshan-Lindblad (GKSL) master equation (Lindblad,
Commun. Math. Phys. 48, 119 (1976); Gorini, Kossakowski and Sudarshan,
J. Math. Phys. 17, 821 (1976)), in units :math:`\hbar = 1`,

.. math::

    \frac{d\rho}{dt} = -i[H, \rho]
        + \sum_k \left( L_k \rho L_k^\dagger
        - \tfrac12 \{ L_k^\dagger L_k, \rho \} \right),

the most general generator of a completely positive, trace-preserving
semigroup. The same dynamics can be *unravelled* into stochastic pure-state
quantum trajectories (the Monte Carlo wavefunction method of Dalibard,
Castin and Molmer, Phys. Rev. Lett. 68, 580 (1992)), whose ensemble average
reproduces :math:`\rho(t)`.

Qubit conventions follow Nielsen and Chuang, *Quantum Computation and
Quantum Information* (2000), Sec. 8.3: :math:`|0\rangle` is the ground
state, :math:`|1\rangle` the excited state, and :func:`sigma_minus`
:math:`= |0\rangle\langle 1|` lowers the excitation.

- :func:`lindblad_rhs`, :func:`lindblad_superoperator`,
  :func:`solve_lindblad`, :func:`lindblad_steady_state` -- the master
  equation and its exact propagation for time-independent generators.
- :class:`QuantumTrajectories`, :class:`TrajectoryResult` -- the Monte
  Carlo wavefunction unravelling.
- :func:`amplitude_damping_kraus`, :func:`dephasing_kraus`,
  :func:`apply_kraus`, :func:`is_trace_preserving` -- the qubit noise
  channels in Kraus form.
- :func:`sigma_minus`, :func:`t1_t2_collapse_operators` -- the collapse
  operators of a qubit with given :math:`T_1` and :math:`T_2`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import expm, null_space

__all__ = [
    "sigma_minus",
    "t1_t2_collapse_operators",
    "lindblad_rhs",
    "lindblad_superoperator",
    "solve_lindblad",
    "lindblad_steady_state",
    "TrajectoryResult",
    "QuantumTrajectories",
    "amplitude_damping_kraus",
    "dephasing_kraus",
    "apply_kraus",
    "is_trace_preserving",
]


def _as_ops(c_ops: Sequence[np.ndarray] | None) -> list[np.ndarray]:
    return [np.asarray(L, dtype=complex) for L in (c_ops or [])]


def sigma_minus() -> np.ndarray:
    r"""Qubit lowering operator :math:`\sigma_- = |0\rangle\langle 1|`.

    Returns
    -------
    numpy.ndarray
        Shape ``(2, 2)``, complex.

    Examples
    --------
    >>> import numpy as np
    >>> sigma_minus() @ np.array([0, 1])  # |1> -> |0>
    array([1.+0.j, 0.+0.j])
    """
    return np.array([[0, 1], [0, 0]], dtype=complex)


def t1_t2_collapse_operators(T1: float, T2: float) -> list[np.ndarray]:
    r"""Collapse operators giving a qubit energy-relaxation time :math:`T_1` and coherence time :math:`T_2`.

    Relaxation :math:`L_1 = \sqrt{1/T_1}\,\sigma_-` damps the excited
    population as :math:`e^{-t/T_1}` and, on its own, the coherences as
    :math:`e^{-t/(2T_1)}`. The remaining decay comes from pure dephasing
    :math:`L_\varphi = \sqrt{\gamma_\varphi/2}\,\sigma_z` with

    .. math::

        \gamma_\varphi = \frac{1}{T_2} - \frac{1}{2T_1},

    so that :math:`|\rho_{01}(t)| = |\rho_{01}(0)|\,e^{-t/T_2}` (Breuer
    and Petruccione, *The Theory of Open Quantum Systems*, 2002, Sec. 3.4).
    Physical rates need :math:`T_2 \le 2T_1`.

    Parameters
    ----------
    T1 : float
        Energy-relaxation time, :math:`T_1 > 0`.
    T2 : float
        Coherence time, :math:`0 < T_2 \le 2T_1`.

    Returns
    -------
    list of numpy.ndarray
        ``[L_1]`` if :math:`T_2 = 2T_1` (no pure dephasing), otherwise
        ``[L_1, L_phi]``.

    Raises
    ------
    ValueError
        If :math:`T_2 > 2T_1` or either time is not positive.

    Examples
    --------
    >>> ops = t1_t2_collapse_operators(T1=1.0, T2=2.0)
    >>> len(ops)
    1
    """
    if T1 <= 0 or T2 <= 0:
        raise ValueError("T1 and T2 must be positive")
    gamma_phi = 1.0 / T2 - 0.5 / T1
    if gamma_phi < -1e-12:
        raise ValueError("unphysical rates: T2 must satisfy T2 <= 2*T1")
    ops = [np.sqrt(1.0 / T1) * sigma_minus()]
    if gamma_phi > 1e-12:
        ops.append(np.sqrt(gamma_phi / 2.0) * np.diag([1.0, -1.0]).astype(complex))
    return ops


def lindblad_rhs(rho: np.ndarray, H: np.ndarray, c_ops: Sequence[np.ndarray] | None = None) -> np.ndarray:
    r"""Right-hand side of the Lindblad master equation, :math:`d\rho/dt`.

    .. math::

        \mathcal{L}\rho = -i[H, \rho]
            + \sum_k \left( L_k \rho L_k^\dagger
            - \tfrac12 L_k^\dagger L_k \rho - \tfrac12 \rho L_k^\dagger L_k \right)

    (Lindblad 1976, Eq. 5.1, with :math:`\hbar = 1`).

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix, shape ``(d, d)``.
    H : numpy.ndarray
        Hamiltonian, shape ``(d, d)``.
    c_ops : sequence of numpy.ndarray, optional
        Collapse (jump) operators :math:`L_k`, rates absorbed into them.

    Returns
    -------
    numpy.ndarray
        :math:`d\rho/dt`, shape ``(d, d)``.

    Examples
    --------
    Spontaneous decay drains the excited state at rate :math:`\gamma`:

    >>> import numpy as np
    >>> rho_e = np.diag([0.0, 1.0]).astype(complex)
    >>> drho = lindblad_rhs(rho_e, np.zeros((2, 2)), [np.sqrt(0.5) * sigma_minus()])
    >>> np.real(np.diag(drho))
    array([ 0.5, -0.5])
    """
    rho = np.asarray(rho, dtype=complex)
    H = np.asarray(H, dtype=complex)
    out = -1j * (H @ rho - rho @ H)
    for L in _as_ops(c_ops):
        LdL = L.conj().T @ L
        out += L @ rho @ L.conj().T - 0.5 * (LdL @ rho + rho @ LdL)
    return out


def lindblad_superoperator(H: np.ndarray, c_ops: Sequence[np.ndarray] | None = None) -> np.ndarray:
    r"""The Liouvillian :math:`\mathcal{L}` as a :math:`d^2 \times d^2` matrix acting on row-major :math:`\mathrm{vec}(\rho)`.

    Uses :math:`\mathrm{vec}(A\rho B) = (A \otimes B^{T})\,\mathrm{vec}(\rho)`
    for the row-major (C-order) flattening ``rho.reshape(-1)``, so

    .. math::

        \mathcal{L} = -i\,(H \otimes I - I \otimes H^{T})
            + \sum_k \left( L_k \otimes L_k^{*}
            - \tfrac12 L_k^\dagger L_k \otimes I
            - \tfrac12 I \otimes (L_k^\dagger L_k)^{T} \right).

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian, shape ``(d, d)``.
    c_ops : sequence of numpy.ndarray, optional
        Collapse operators.

    Returns
    -------
    numpy.ndarray
        Shape ``(d**2, d**2)``, complex.

    Examples
    --------
    Its action on ``rho.reshape(-1)`` agrees with :func:`lindblad_rhs`:

    >>> import numpy as np
    >>> H = np.array([[0.0, 0.3], [0.3, 1.0]])
    >>> ops = [sigma_minus()]
    >>> rho = np.array([[0.4, 0.1j], [-0.1j, 0.6]])
    >>> S = lindblad_superoperator(H, ops)
    >>> np.allclose(S @ rho.reshape(-1), lindblad_rhs(rho, H, ops).reshape(-1))
    True
    """
    H = np.asarray(H, dtype=complex)
    d = H.shape[0]
    eye = np.eye(d, dtype=complex)
    S = -1j * (np.kron(H, eye) - np.kron(eye, H.T))
    for L in _as_ops(c_ops):
        LdL = L.conj().T @ L
        S += np.kron(L, L.conj()) - 0.5 * np.kron(LdL, eye) - 0.5 * np.kron(eye, LdL.T)
    return S


def solve_lindblad(
    rho0: np.ndarray,
    H: np.ndarray,
    c_ops: Sequence[np.ndarray] | None,
    times: np.ndarray,
) -> np.ndarray:
    r"""Propagate a density matrix under a time-independent Lindblad equation.

    The solution is exact up to round-off: :math:`\mathrm{vec}\,\rho(t) =
    e^{\mathcal{L}(t - t_0)}\,\mathrm{vec}\,\rho(t_0)`, with the matrix
    exponential of :func:`lindblad_superoperator` taken once per distinct
    time step (a uniform grid costs a single :func:`scipy.linalg.expm`).

    Parameters
    ----------
    rho0 : numpy.ndarray
        Initial density matrix at ``times[0]``, shape ``(d, d)``. A state
        vector of shape ``(d,)`` is accepted and converted to
        :math:`|\psi\rangle\langle\psi|`.
    H : numpy.ndarray
        Hamiltonian, shape ``(d, d)``.
    c_ops : sequence of numpy.ndarray or None
        Collapse operators.
    times : numpy.ndarray
        Increasing output times.

    Returns
    -------
    numpy.ndarray
        :math:`\rho(t)` at each time, shape ``(len(times), d, d)``.

    Examples
    --------
    :math:`T_1` decay of the excited state, :math:`P_1(t) = e^{-t/T_1}`:

    >>> import numpy as np
    >>> t = np.array([0.0, 1.0, 2.0])
    >>> rho = solve_lindblad(np.array([0, 1]), np.zeros((2, 2)), [sigma_minus()], t)
    >>> np.allclose(rho[:, 1, 1].real, np.exp(-t))
    True
    """
    rho0 = np.asarray(rho0, dtype=complex)
    if rho0.ndim == 1:
        rho0 = np.outer(rho0, rho0.conj())
    times = np.asarray(times, dtype=float)
    d = rho0.shape[0]
    S = lindblad_superoperator(H, c_ops)
    out = np.empty((len(times), d, d), dtype=complex)
    vec = rho0.reshape(-1).copy()
    out[0] = rho0
    cache: dict[float, np.ndarray] = {}
    for i in range(1, len(times)):
        dt = round(float(times[i] - times[i - 1]), 12)
        if dt not in cache:
            cache[dt] = expm(S * dt)
        vec = cache[dt] @ vec
        out[i] = vec.reshape(d, d)
    return out


def lindblad_steady_state(H: np.ndarray, c_ops: Sequence[np.ndarray] | None) -> np.ndarray:
    r"""The stationary state :math:`\mathcal{L}\rho_{ss} = 0`, normalized to unit trace.

    Solved as the null space of :func:`lindblad_superoperator`. For a
    unique steady state the null space is one-dimensional.

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian, shape ``(d, d)``.
    c_ops : sequence of numpy.ndarray or None
        Collapse operators.

    Returns
    -------
    numpy.ndarray
        :math:`\rho_{ss}`, shape ``(d, d)``.

    Raises
    ------
    ValueError
        If the steady state is not unique.

    Examples
    --------
    Resonance fluorescence: a qubit driven at Rabi frequency :math:`\Omega`
    and decaying at rate :math:`\gamma` settles to excited population
    :math:`\Omega^2/(\gamma^2 + 2\Omega^2)` (Loudon, *The Quantum Theory
    of Light*, 3rd ed., Eq. 2.8.21 on resonance):

    >>> import numpy as np
    >>> Omega, gamma = 1.0, 0.5
    >>> H = 0.5 * Omega * np.array([[0, 1], [1, 0]])
    >>> rho = lindblad_steady_state(H, [np.sqrt(gamma) * sigma_minus()])
    >>> round(float(rho[1, 1].real), 10) == round(Omega**2 / (gamma**2 + 2 * Omega**2), 10)
    True
    """
    S = lindblad_superoperator(H, c_ops)
    d = int(round(np.sqrt(S.shape[0])))
    ns = null_space(S, rcond=1e-10)
    if ns.shape[1] != 1:
        raise ValueError(f"steady state is not unique (null space has dimension {ns.shape[1]})")
    rho = ns[:, 0].reshape(d, d)
    rho = rho / np.trace(rho)
    return 0.5 * (rho + rho.conj().T)


# --- Monte Carlo wavefunction (quantum-trajectory) unravelling ----------------


@dataclass
class TrajectoryResult:
    r"""Output of :meth:`QuantumTrajectories.run`.

    Attributes
    ----------
    times : numpy.ndarray
        Output times, shape ``(n_t,)``.
    states : numpy.ndarray
        Normalized state of every trajectory at every output time, shape
        ``(n_traj, n_t, d)``.
    jump_times : list of numpy.ndarray
        For each trajectory, the times at which a quantum jump occurred.
    jump_channels : list of numpy.ndarray
        For each trajectory, the index :math:`k` of the collapse operator
        :math:`L_k` applied at each jump.
    """

    times: np.ndarray
    states: np.ndarray
    jump_times: list = field(default_factory=list)
    jump_channels: list = field(default_factory=list)

    @property
    def n_trajectories(self) -> int:
        """Number of trajectories."""
        return self.states.shape[0]

    def density_matrices(self) -> np.ndarray:
        r"""Ensemble-averaged density matrix :math:`\overline{|\psi\rangle\langle\psi|}` at each time.

        Returns
        -------
        numpy.ndarray
            Shape ``(n_t, d, d)``.
        """
        return np.einsum("rti,rtj->tij", self.states, self.states.conj()) / self.n_trajectories

    def expectation(self, op: np.ndarray, average: bool = True) -> np.ndarray:
        r""":math:`\langle\psi|A|\psi\rangle` along each trajectory, or its ensemble average.

        Parameters
        ----------
        op : numpy.ndarray
            Operator :math:`A`, shape ``(d, d)``.
        average : bool, default=True
            If True, average over trajectories.

        Returns
        -------
        numpy.ndarray
            Real part of the expectation value, shape ``(n_t,)`` if
            ``average`` else ``(n_traj, n_t)``.
        """
        vals = np.einsum("rti,ij,rtj->rt", self.states.conj(), np.asarray(op, dtype=complex), self.states).real
        return vals.mean(axis=0) if average else vals


@dataclass
class QuantumTrajectories:
    r"""Monte Carlo wavefunction (quantum-jump) unravelling of a Lindblad equation.

    Each trajectory evolves an unnormalized state under the non-Hermitian
    effective Hamiltonian

    .. math::

        H_\mathrm{eff} = H - \frac{i}{2} \sum_k L_k^\dagger L_k,

    whose squared norm decays from 1. A uniform random number :math:`r` is
    drawn, and when :math:`\lVert\psi\rVert^2` falls below :math:`r` a
    quantum jump :math:`\psi \to L_k\psi` is applied, channel :math:`k`
    chosen with probability :math:`\propto \lVert L_k \psi \rVert^2`; the
    state is renormalized and a fresh :math:`r` drawn (the waiting-time
    formulation of Dalibard, Castin and Molmer 1992 and Dum, Zoller and
    Ritsch, Phys. Rev. A 45, 4879 (1992); review: Plenio and Knight, Rev.
    Mod. Phys. 70, 101 (1998), Sec. III). The ensemble average of
    :math:`|\psi\rangle\langle\psi|` obeys the Lindblad equation.

    The no-jump propagator :math:`e^{-iH_\mathrm{eff}\,dt}` is applied
    exactly, so the only discretization error is the :math:`O(dt)`
    resolution of each jump time. All trajectories are advanced together
    as one vectorized array.

    Parameters
    ----------
    H : numpy.ndarray
        Hamiltonian, shape ``(d, d)``.
    c_ops : sequence of numpy.ndarray
        Collapse operators :math:`L_k`.

    Examples
    --------
    Starting in the excited state with pure decay, every trajectory
    eventually makes exactly one jump:

    >>> import numpy as np
    >>> traj = QuantumTrajectories(np.zeros((2, 2)), [sigma_minus()])
    >>> res = traj.run(np.array([0, 1]), np.linspace(0, 20, 5), n_traj=50, dt=0.01,
    ...                rng=np.random.default_rng(1))
    >>> {len(j) for j in res.jump_times}
    {1}
    """

    H: np.ndarray
    c_ops: Sequence[np.ndarray]

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=complex)
        self.c_ops = _as_ops(self.c_ops)

    @property
    def effective_hamiltonian(self) -> np.ndarray:
        r"""The non-Hermitian :math:`H_\mathrm{eff} = H - \frac{i}{2}\sum_k L_k^\dagger L_k`."""
        H_eff = self.H.copy()
        for L in self.c_ops:
            H_eff -= 0.5j * L.conj().T @ L
        return H_eff

    def run(
        self,
        psi0: np.ndarray,
        times: np.ndarray,
        n_traj: int = 500,
        dt: float | None = None,
        rng: np.random.Generator | None = None,
    ) -> TrajectoryResult:
        r"""Generate an ensemble of quantum trajectories.

        Parameters
        ----------
        psi0 : numpy.ndarray
            Initial pure state, shape ``(d,)``; normalized internally.
        times : numpy.ndarray
            Increasing output times; ``times[0]`` is the start time.
        n_traj : int, default=500
            Number of trajectories.
        dt : float, optional
            Maximum internal step. Defaults to one hundredth of the fastest
            jump timescale :math:`1/\sum_k \lVert L_k^\dagger L_k\rVert`,
            capped by the output spacing.
        rng : numpy.random.Generator, optional
            Random generator (for reproducibility).

        Returns
        -------
        TrajectoryResult
        """
        rng = np.random.default_rng() if rng is None else rng
        times = np.asarray(times, dtype=float)
        psi0 = np.asarray(psi0, dtype=complex)
        psi0 = psi0 / np.linalg.norm(psi0)
        d = psi0.shape[0]
        if dt is None:
            total_rate = sum(np.linalg.norm(L.conj().T @ L, 2) for L in self.c_ops)
            dt = 0.01 / total_rate if total_rate > 0 else np.inf
        H_eff = self.effective_hamiltonian

        psi = np.tile(psi0, (n_traj, 1))
        r = rng.random(n_traj)
        states = np.empty((n_traj, len(times), d), dtype=complex)
        states[:, 0] = psi0
        jump_times: list[list[float]] = [[] for _ in range(n_traj)]
        jump_channels: list[list[int]] = [[] for _ in range(n_traj)]
        cache: dict[float, np.ndarray] = {}

        t = times[0]
        for i in range(1, len(times)):
            span = times[i] - times[i - 1]
            n_sub = max(1, int(np.ceil(span / dt - 1e-9)))
            h = round(span / n_sub, 14)
            if h not in cache:
                cache[h] = expm(-1j * H_eff * h).T  # row-vector convention: psi @ U^T
            U_T = cache[h]
            for _ in range(n_sub):
                psi = psi @ U_T
                t += h
                norm2 = np.einsum("ri,ri->r", psi.conj(), psi).real
                jumpers = np.nonzero(norm2 <= r)[0]
                if jumpers.size and self.c_ops:
                    candidates = np.stack([psi[jumpers] @ L.T for L in self.c_ops])  # (K, J, d)
                    weights = np.einsum("kji,kji->jk", candidates.conj(), candidates).real
                    cum = np.cumsum(weights, axis=1)
                    pick = (rng.random(jumpers.size) * cum[:, -1])[:, None]
                    channel = np.minimum((cum < pick).sum(axis=1), len(self.c_ops) - 1)
                    new = candidates[channel, np.arange(jumpers.size)]
                    psi[jumpers] = new / np.linalg.norm(new, axis=1, keepdims=True)
                    r[jumpers] = rng.random(jumpers.size)
                    for j, k in zip(jumpers, channel):
                        jump_times[j].append(t)
                        jump_channels[j].append(int(k))
            states[:, i] = psi / np.linalg.norm(psi, axis=1, keepdims=True)

        return TrajectoryResult(
            times=times,
            states=states,
            jump_times=[np.array(j) for j in jump_times],
            jump_channels=[np.array(c, dtype=int) for c in jump_channels],
        )


# --- Quantum channels in Kraus form -----------------------------------------


def amplitude_damping_kraus(gamma: float) -> list[np.ndarray]:
    r"""Kraus operators of the qubit amplitude-damping channel.

    .. math::

        E_0 = \begin{pmatrix} 1 & 0 \\ 0 & \sqrt{1-\gamma} \end{pmatrix},
        \qquad
        E_1 = \begin{pmatrix} 0 & \sqrt{\gamma} \\ 0 & 0 \end{pmatrix}

    (Nielsen and Chuang, Eq. 8.108): the excited state :math:`|1\rangle`
    decays to :math:`|0\rangle` with probability :math:`\gamma`. Lindblad
    decay at rate :math:`\Gamma` for time :math:`t` is exactly this channel
    with :math:`\gamma = 1 - e^{-\Gamma t}`.

    Parameters
    ----------
    gamma : float
        Decay probability, :math:`0 \le \gamma \le 1`.

    Returns
    -------
    list of numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> rho = apply_kraus(np.diag([0.0, 1.0]), amplitude_damping_kraus(0.3))
    >>> np.real(np.diag(rho)).round(3)
    array([0.3, 0.7])
    """
    return [
        np.array([[1, 0], [0, np.sqrt(1 - gamma)]], dtype=complex),
        np.array([[0, np.sqrt(gamma)], [0, 0]], dtype=complex),
    ]


def dephasing_kraus(p: float) -> list[np.ndarray]:
    r"""Kraus operators of the qubit dephasing (phase-flip) channel.

    .. math::

        E_0 = \sqrt{1-p}\,I, \qquad E_1 = \sqrt{p}\,\sigma_z

    (Nielsen and Chuang, Eq. 8.3.3): populations are untouched and the
    coherences shrink by :math:`1 - 2p`. Pure Lindblad dephasing
    :math:`L = \sqrt{\gamma_\varphi/2}\,\sigma_z` for time :math:`t` is this
    channel with :math:`p = (1 - e^{-\gamma_\varphi t})/2`.

    Parameters
    ----------
    p : float
        Phase-flip probability, :math:`0 \le p \le 1`.

    Returns
    -------
    list of numpy.ndarray

    Examples
    --------
    >>> import numpy as np
    >>> plus = 0.5 * np.ones((2, 2))
    >>> round(float(apply_kraus(plus, dephasing_kraus(0.1))[0, 1].real), 12)
    0.4
    """
    return [np.sqrt(1 - p) * np.eye(2, dtype=complex), np.sqrt(p) * np.diag([1.0, -1.0]).astype(complex)]


def apply_kraus(rho: np.ndarray, kraus: Sequence[np.ndarray]) -> np.ndarray:
    r"""Apply a channel in Kraus form, :math:`\rho \mapsto \sum_i E_i \rho E_i^\dagger`.

    Parameters
    ----------
    rho : numpy.ndarray
        Density matrix, shape ``(d, d)``.
    kraus : sequence of numpy.ndarray
        Kraus operators :math:`E_i`.

    Returns
    -------
    numpy.ndarray
        The output density matrix.

    Examples
    --------
    >>> import numpy as np
    >>> rho = np.diag([0.5, 0.5]).astype(complex)
    >>> np.allclose(apply_kraus(rho, dephasing_kraus(0.2)), rho)
    True
    """
    rho = np.asarray(rho, dtype=complex)
    return sum(E @ rho @ E.conj().T for E in _as_ops(kraus))


def is_trace_preserving(kraus: Sequence[np.ndarray], atol: float = 1e-12) -> bool:
    r"""Check the completeness relation :math:`\sum_i E_i^\dagger E_i = I`.

    Parameters
    ----------
    kraus : sequence of numpy.ndarray
        Kraus operators.
    atol : float, default=1e-12
        Absolute tolerance.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_trace_preserving(amplitude_damping_kraus(0.4))
    True
    """
    ops = _as_ops(kraus)
    total = sum(E.conj().T @ E for E in ops)
    return bool(np.allclose(total, np.eye(ops[0].shape[0]), atol=atol))
