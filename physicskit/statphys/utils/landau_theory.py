"""Landau's phenomenological theory of continuous phase transitions.

Rather than derive a phase transition from microscopic interactions, Landau
(1937) asked what the *free energy itself* must look like, given only the
symmetry of the order parameter, near a continuous transition. For an
order parameter :math:`m` (magnetization, density difference, ...) that
the disordered phase's symmetry forbids from appearing at odd order, the
simplest possible expansion is

.. math::

    F(m, T) = a(T - T_C)\\, m^2 + b\\, m^4 - h\\, m, \\qquad a, b > 0,

with an external field :math:`h` coupling linearly to :math:`m`. Minimizing
this quartic over :math:`m` reproduces, from symmetry and analyticity
alone, the qualitative shape of every continuous transition: a single
minimum at :math:`m = 0` above :math:`T_C`, a spontaneous, symmetry-broken
pair of minima :math:`m = \\pm\\sqrt{a(T_C - T)/2b}` below it, and a
susceptibility that diverges from both sides as :math:`T \\to T_C`. Landau
theory's mean-field critical exponents (:math:`\\beta = 1/2`, :math:`\\gamma
= 1`) are quantitatively wrong in low dimensions -- famously not matching
Onsager's exact 2D Ising values -- but the qualitative order-parameter
picture it introduced, and the idea of classifying transitions by how they
break symmetry, underlies the entire modern theory of phase transitions
that the renormalization group later completed.
"""

from __future__ import annotations

import numpy as np

__all__ = ["landau_equilibrium_magnetization", "landau_free_energy", "landau_susceptibility"]


def landau_free_energy(m, T, Tc, a=1.0, b=1.0, h=0.0):
    """Landau free energy :math:`F(m) = a(T - T_C) m^2 + b m^4 - h m`.

    Parameters
    ----------
    m : array_like
        Order parameter value(s) at which to evaluate :math:`F`.
    T : float
        Temperature.
    Tc : float
        Critical temperature.
    a, b : float, default=1.0
        Positive Landau expansion coefficients.
    h : float, default=0.0
        External field, linearly coupled to :math:`m`.

    Returns
    -------
    ndarray
    """
    m = np.asarray(m, dtype=np.float64)
    return a * (T - Tc) * m**2 + b * m**4 - h * m


def landau_equilibrium_magnetization(T, Tc, a=1.0, b=1.0, h=0.0):
    """Order parameter minimizing the Landau free energy at temperature T.

    Solves :math:`\\partial F/\\partial m = 2a(T-T_C)m + 4bm^3 - h = 0` for
    every real root, and returns the one with the lowest :math:`F` (the
    true equilibrium branch, correctly following the symmetry-broken
    minimum rather than the unstable one once :math:`h \\ne 0` tilts the
    double well). At :math:`h = 0` below :math:`T_C` the two symmetry-broken
    roots :math:`\\pm\\sqrt{a(T_C-T)/2b}` are exactly degenerate in
    :math:`F`; ties are broken deterministically in favor of the
    non-negative root (the conventional "up" branch of spontaneous symmetry
    breaking), so that :math:`m(T)` is a single-valued, continuous curve
    rather than jumping unpredictably between the two equally valid
    branches as :math:`T` is swept. At :math:`h = 0` this reduces to the
    textbook result :math:`m = 0` for :math:`T \\ge T_C` and :math:`m =
    \\sqrt{a(T_C - T)/2b}` for :math:`T < T_C` -- the mean-field
    :math:`\\beta = 1/2` critical exponent.

    Parameters
    ----------
    T : float or array_like
        Temperature(s).
    Tc : float
        Critical temperature.
    a, b : float, default=1.0
        Positive Landau expansion coefficients.
    h : float, default=0.0
        External field.

    Returns
    -------
    float or ndarray
        Equilibrium order parameter, matching the shape of ``T``.

    Examples
    --------
    >>> round(landau_equilibrium_magnetization(T=1.0, Tc=2.0, a=1.0, b=1.0, h=0.0), 6)
    0.707107
    >>> landau_equilibrium_magnetization(T=3.0, Tc=2.0, a=1.0, b=1.0, h=0.0)
    0.0
    """
    T_arr = np.atleast_1d(np.asarray(T, dtype=np.float64))
    m_eq = np.empty_like(T_arr)
    for i, Ti in enumerate(T_arr):
        coeffs = [4.0 * b, 0.0, 2.0 * a * (Ti - Tc), -h]
        roots = np.roots(coeffs)
        real_roots = roots[np.abs(roots.imag) < 1e-8].real
        F_vals = a * (Ti - Tc) * real_roots**2 + b * real_roots**4 - h * real_roots
        # Below T_C at h=0 the +/-m branches are exactly degenerate in F, so
        # argmin alone would tie-break on np.roots' arbitrary root ordering,
        # flipping sign unpredictably as T is swept. Restrict to the
        # (numerically) tied minimizers and deterministically prefer the
        # largest (non-negative, "up"-branch) one among them.
        is_min = F_vals <= F_vals.min() + 1e-9 * (1.0 + np.abs(F_vals.min()))
        m_eq[i] = real_roots[is_min].max()
    return float(m_eq[0]) if np.ndim(T) == 0 else m_eq


def landau_susceptibility(T, Tc, a=1.0, b=1.0):
    """Zero-field susceptibility :math:`\\chi = (\\partial^2 F/\\partial m^2)^{-1}` at equilibrium.

    Evaluated at the :math:`h=0` equilibrium magnetization, this gives the
    standard mean-field result

    .. math::

        \\chi(T) = \\begin{cases}
            \\dfrac{1}{2a(T - T_C)}, & T > T_C, \\\\[4pt]
            \\dfrac{1}{4a(T_C - T)}, & T < T_C,
        \\end{cases}

    diverging on both sides of :math:`T_C` with the same exponent
    :math:`\\gamma = 1` but different amplitudes -- the universal
    mean-field amplitude ratio of 2.

    Parameters
    ----------
    T : array_like
        Temperature(s); may include ``Tc`` itself, where :math:`\\chi`
        diverges (``inf`` is returned there).
    Tc : float
        Critical temperature.
    a, b : float, default=1.0
        Positive Landau expansion coefficients (``b`` is unused above
        :math:`T_C`, kept for a uniform signature).

    Returns
    -------
    ndarray
    """
    T = np.asarray(T, dtype=np.float64)
    with np.errstate(divide="ignore"):
        return np.where(T > Tc, 1.0 / (2.0 * a * (T - Tc)), 1.0 / (4.0 * a * (Tc - T)))
