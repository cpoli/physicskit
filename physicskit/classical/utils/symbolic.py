"""SymPy engine that auto-derives equations of motion from a Lagrangian.

Given an arbitrary Lagrangian L(q, qdot, t), :class:`LagrangianEngine`:

1. Forms the Euler-Lagrange equations
   d/dt(dL/dqdot_i) - dL/dq_i = 0
   which are linear in qddot, and solves them for the generalized
   accelerations qddot_i (equivalently: derives the mass matrix M(q)
   and inverts it).
2. Performs the Legendre transform p_i = dL/dqdot_i and derives the
   Hamiltonian H(q, p, t) = sum_i p_i qdot_i - L, giving dH/dq and dH/dp
   in closed form. This is what makes the *implicit midpoint* symplectic
   integrator applicable even when M(q) is configuration-dependent and
   the system is therefore non-separable (e.g. the double pendulum).
3. Lambdifies every derived vector expression and JIT-compiles it with
   Numba, exposing plain ``@njit`` dispatchers so the equations of
   motion run at native speed inside :mod:`physicskit.classical.core.integrators`
   (whose step functions call these callbacks from nopython code, which
   requires them to be real njit dispatchers rather than closures over
   Python objects).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import sympy as sp
from numba import njit

__all__ = ["LagrangianEngine"]


def _lambdify_vector_njit(args: Sequence[sp.Symbol], exprs: Sequence[sp.Expr]):
    """Lambdify a list of scalar expressions and JIT-compile it.

    Dynamically generated functions cannot use numba's on-disk cache.

    Parameters
    ----------
    args : sequence of sympy Symbol
        Positional arguments of the generated function.
    exprs : sequence of sympy expression
        Scalar expressions to evaluate, in order.

    Returns
    -------
    callable
        An ``@njit`` dispatcher returning a Python list of scalars.
    """
    raw = sp.lambdify(tuple(args), list(exprs), modules="numpy")
    return njit(cache=False)(raw)


def _indexed_wrapper(raw_jit, n: int, as_array: bool = True):
    """Build an ``@njit`` wrapper ``f(arr1, arr2, scalar) -> ...`` around a lambdified function.

    The wrapped function's signature is ``(arr1[0..n-1], arr2[0..n-1], scalar)``.
    Numba cannot star-unpack a runtime numpy array in nopython mode
    (the argument count must be known at compile time), so this
    generates explicit-index source (``arr1[0], arr1[1], ...``) via
    ``exec`` -- the same technique SymPy's own ``lambdify`` uses -- and
    then JIT-compiles it. ``n`` is fixed per system (a compile-time
    constant), so this runs once, at system-construction time.

    Parameters
    ----------
    raw_jit : callable
        An ``@njit`` dispatcher taking ``2 * n + 1`` scalar arguments.
    n : int
        Length of each array argument.
    as_array : bool
        If True, wrap the result in ``np.array(...)``; if False, return
        it as-is (for scalar-valued ``raw_jit``, e.g. a Hamiltonian).

    Returns
    -------
    callable
        An ``@njit`` dispatcher ``(arr1, arr2, scalar) -> result``.
    """
    call_args = [f"a1[{i}]" for i in range(n)] + [f"a2[{i}]" for i in range(n)] + ["s"]
    body = f"np.array(raw_jit({', '.join(call_args)}))" if as_array else f"raw_jit({', '.join(call_args)})"
    src = f"def _wrapper(a1, a2, s):\n    return {body}\n"
    ns = {"raw_jit": raw_jit, "np": np}
    exec(src, ns)
    return njit(cache=False)(ns["_wrapper"])


class LagrangianEngine:
    """Derive and JIT-compile equations of motion from L(q, qdot, t).

    Parameters
    ----------
    q, qdot : sequences of sympy Symbol
        Generalized coordinates and velocities, same length/order.
    L : sympy expression
        The Lagrangian, a function of ``q``, ``qdot``, optionally the
        time symbol ``t``, and any constant parameters (substituted via
        ``params`` before compilation).
    t : sympy Symbol, optional
        Time symbol appearing in ``L`` (e.g. for a rotating hoop with
        explicit time dependence). A fresh dummy symbol is used if
        omitted.
    params : dict[sympy.Symbol, float], optional
        Numeric values substituted for constant parameters in ``L``.

    Attributes
    ----------
    acceleration_njit : callable
        ``(q, qdot, t) -> qddot``.
    momentum_njit : callable
        ``(q, qdot) -> p``, where ``p_i = dL/dqdot_i``.
    canonical_deriv_njit : callable
        ``(t, y) -> dy``, where ``y = [q, p]`` and ``dy = [dH/dp, -dH/dq]``.
    velocity_njit : callable
        ``(q, p) -> qdot``, i.e. ``dH/dp``.
    full_deriv_njit : callable
        ``(t, y) -> dy``, where ``y = [q, qdot]`` and ``dy = [qdot, qddot]``.
    hamiltonian_njit : callable
        ``(q, p, t) -> H``.
    mass_matrix : sympy.Matrix
        The symbolic mass matrix M(q) from the Euler-Lagrange derivation.
    H_expr : sympy expression
        The derived Hamiltonian H(q, p, t).
    """

    def __init__(self, q, qdot, L: sp.Expr, t: sp.Symbol = None, params: dict = None):
        self.q = list(q)
        self.qdot = list(qdot)
        self.n = len(self.q)
        self.t = t if t is not None else sp.Symbol("t")
        self.params = params or {}
        self.L = sp.sympify(L).subs(self.params)

        self._qddot = sp.symbols(f"qddot0:{self.n}")
        self._p = sp.symbols(f"p0:{self.n}")
        self._derive_euler_lagrange()
        self._derive_hamiltonian()
        self._compile()

    # -- Euler-Lagrange -----------------------------------------------
    def _derive_euler_lagrange(self):
        q, qdot, qddot, t = self.q, self.qdot, self._qddot, self.t
        n = self.n

        dL_dqdot = [sp.diff(self.L, qdot[i]) for i in range(n)]
        eqs = []
        for i in range(n):
            total_ddt = sp.diff(dL_dqdot[i], t)
            for j in range(n):
                total_ddt += sp.diff(dL_dqdot[i], q[j]) * qdot[j]
                total_ddt += sp.diff(dL_dqdot[i], qdot[j]) * qddot[j]
            eqs.append(sp.together(total_ddt - sp.diff(self.L, q[i])))

        M, rhs = sp.linear_eq_to_matrix(eqs, list(qddot))
        self.mass_matrix = sp.simplify(M)
        self._qddot_expr = sp.simplify(self.mass_matrix.inv() * rhs)

    # -- Legendre transform / Hamiltonian ------------------------------
    def _derive_hamiltonian(self):
        q, qdot, p = self.q, self.qdot, self._p
        n = self.n

        dL_dqdot = [sp.diff(self.L, qdot[i]) for i in range(n)]
        A, b = sp.linear_eq_to_matrix([dL_dqdot[i] - p[i] for i in range(n)], list(qdot))
        qdot_of_p = A.inv() * b
        qdot_subs = {qdot[i]: qdot_of_p[i] for i in range(n)}

        H = sum(p[i] * qdot_of_p[i] for i in range(n)) - self.L.subs(qdot_subs, simultaneous=True)
        self.H_expr = sp.simplify(H)

        self._dHdq = [sp.simplify(sp.diff(self.H_expr, q[i])) for i in range(n)]
        self._dHdp = [sp.simplify(sp.diff(self.H_expr, p[i])) for i in range(n)]
        self._velocity_expr = [sp.simplify(qdot_of_p[i]) for i in range(n)]
        self._p_of_qdot_expr = list(dL_dqdot)

    # -- compile everything to njit callables --------------------------
    def _compile(self):
        q, qdot, p, t = self.q, self.qdot, self._p, self.t
        n = self.n
        qp_args = tuple(q) + tuple(qdot) + (t,)
        qP_args = tuple(q) + tuple(p) + (t,)

        accel_raw = _lambdify_vector_njit(qp_args, self._qddot_expr)
        grad_q_raw = _lambdify_vector_njit(qP_args, self._dHdq)
        grad_p_raw = _lambdify_vector_njit(qP_args, self._dHdp)
        vel_raw = _lambdify_vector_njit(qP_args, self._velocity_expr)
        p_raw = _lambdify_vector_njit(qp_args, self._p_of_qdot_expr)
        H_raw = njit(cache=False)(sp.lambdify(qP_args, self.H_expr, modules="numpy"))

        acceleration_njit = _indexed_wrapper(accel_raw, n)
        momentum_njit_full = _indexed_wrapper(p_raw, n)
        grad_q_njit = _indexed_wrapper(grad_q_raw, n)
        grad_p_njit = _indexed_wrapper(grad_p_raw, n)
        velocity_njit_full = _indexed_wrapper(vel_raw, n)
        hamiltonian_njit_full = _indexed_wrapper(H_raw, n, as_array=False)

        @njit(cache=False)
        def momentum_njit(qa, qdota):
            return momentum_njit_full(qa, qdota, 0.0)

        @njit(cache=False)
        def canonical_deriv_njit(tt, y):
            qa = y[:n]
            pa = y[n:]
            dHdq = grad_q_njit(qa, pa, tt)
            dHdp = grad_p_njit(qa, pa, tt)
            out = np.empty(2 * n)
            out[:n] = dHdp
            out[n:] = -dHdq
            return out

        @njit(cache=False)
        def velocity_njit(qa, pa):
            return velocity_njit_full(qa, pa, 0.0)

        @njit(cache=False)
        def full_deriv_njit(tt, y):
            qa = y[:n]
            qdota = y[n:]
            out = np.empty(2 * n)
            out[:n] = qdota
            out[n:] = acceleration_njit(qa, qdota, tt)
            return out

        @njit(cache=False)
        def hamiltonian_njit(qa, pa, tt):
            return hamiltonian_njit_full(qa, pa, tt)

        self.acceleration_njit = acceleration_njit
        self.momentum_njit = momentum_njit
        self.canonical_deriv_njit = canonical_deriv_njit
        self.velocity_njit = velocity_njit
        self.full_deriv_njit = full_deriv_njit
        self.hamiltonian_njit = hamiltonian_njit
