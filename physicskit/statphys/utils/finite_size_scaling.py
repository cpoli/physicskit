"""Finite-size scaling analysis: locating T_C and estimating critical exponents.

Monte Carlo simulations necessarily run on finite lattices, so an apparent
thermodynamic singularity is always rounded and shifted relative to its
infinite-volume value. Finite-size scaling theory predicts precisely how
observables at different lattice sizes :math:`L` relate near a continuous
phase transition, letting :math:`T_C` and the critical exponents be
extracted from a family of finite simulations without ever simulating an
infinite system.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "binder_cumulant_crossing",
    "estimate_beta_over_nu",
    "estimate_gamma_over_nu",
    "estimate_nu_from_binder_slope",
    "power_law_exponent",
]


def power_law_exponent(x, y):
    """Fit the exponent of an assumed power law :math:`y = A x^p` via log-log linear regression.

    Parameters
    ----------
    x, y : array_like
        Positive-valued samples believed to follow a power law.

    Returns
    -------
    exponent : float
        The fitted exponent :math:`p`.
    prefactor : float
        The fitted prefactor :math:`A`.

    Examples
    --------
    >>> x = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
    >>> y = 3.0 * x**1.5
    >>> p, A = power_law_exponent(x, y)
    >>> round(p, 6)
    1.5
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    p, log_A = np.polyfit(np.log(x), np.log(y), 1)
    return float(p), float(np.exp(log_A))


def estimate_gamma_over_nu(L_values, chi_max_values):
    """Estimate :math:`\\gamma/\\nu` from the lattice-size scaling of the peak susceptibility.

    Finite-size scaling predicts :math:`\\chi_{\\max}(L) \\sim L^{\\gamma/\\nu}`.

    Parameters
    ----------
    L_values : array_like
        Lattice sizes.
    chi_max_values : array_like
        Peak susceptibility measured at each lattice size (e.g. the maximum
        of the ``"chi"`` array returned by ``Ising2D.run_temperature_sweep``).

    Returns
    -------
    float
        Estimated :math:`\\gamma / \\nu`.
    """
    exponent, _ = power_law_exponent(L_values, chi_max_values)
    return exponent


def estimate_beta_over_nu(L_values, magnetization_at_tc_values):
    """Estimate :math:`\\beta/\\nu` from the lattice-size scaling of the order parameter at T_C.

    Finite-size scaling predicts :math:`M(T_C, L) \\sim L^{-\\beta/\\nu}`.

    Parameters
    ----------
    L_values : array_like
        Lattice sizes.
    magnetization_at_tc_values : array_like
        Order parameter measured at (approximately) :math:`T_C` for each
        lattice size.

    Returns
    -------
    float
        Estimated :math:`\\beta / \\nu`.
    """
    exponent, _ = power_law_exponent(L_values, magnetization_at_tc_values)
    return -exponent


def estimate_nu_from_binder_slope(L_values, T_grids, U4_grids, T_c):
    """Estimate the correlation-length exponent :math:`\\nu` from the Binder cumulant's slope at T_C.

    Finite-size scaling predicts the Binder cumulant's temperature
    derivative at criticality scales as
    :math:`\\left.\\frac{dU_4}{dT}\\right|_{T_C} \\sim L^{1/\\nu}`. This
    numerically differentiates each lattice size's :math:`U_4(T)` curve at
    :math:`T_C` (a finite-difference slope between the two grid points
    bracketing :math:`T_C`) and fits the resulting :math:`L`-scaling.

    Parameters
    ----------
    L_values : array_like
        Lattice sizes.
    T_grids : list of array_like
        Temperature grid used at each lattice size (need not be identical
        across sizes, but each must bracket ``T_c``).
    U4_grids : list of array_like
        Binder cumulant values on each temperature grid.
    T_c : float
        Critical temperature at which to evaluate the slope.

    Returns
    -------
    float
        Estimated :math:`\\nu`.
    """
    slopes = []
    for T, U4 in zip(T_grids, U4_grids):
        T = np.asarray(T, dtype=np.float64)
        U4 = np.asarray(U4, dtype=np.float64)
        order = np.argsort(T)
        T, U4 = T[order], U4[order]
        idx = np.clip(np.searchsorted(T, T_c), 1, len(T) - 1)
        slope = (U4[idx] - U4[idx - 1]) / (T[idx] - T[idx - 1])
        slopes.append(abs(slope))
    exponent, _ = power_law_exponent(L_values, slopes)
    return 1.0 / exponent


def binder_cumulant_crossing(T, U4_by_L):
    """Estimate T_C as the crossing point of the two largest lattice sizes' Binder cumulant curves.

    Interpolates linearly between grid points to find where the two curves
    cross -- the standard, finite-size-correction-robust way to locate a
    continuous transition's critical temperature.

    Parameters
    ----------
    T : array_like
        Common temperature grid shared by all lattice sizes.
    U4_by_L : dict of int -> array_like
        Binder cumulant values at each lattice size, evaluated on ``T``.

    Returns
    -------
    float
        Estimated crossing temperature.

    Raises
    ------
    ValueError
        If the two largest lattice sizes' curves never cross on ``T``.
    """
    L1, L2 = sorted(U4_by_L.keys())[-2:]
    T = np.asarray(T, dtype=np.float64)
    diff = np.asarray(U4_by_L[L1], dtype=np.float64) - np.asarray(U4_by_L[L2], dtype=np.float64)
    sign_changes = np.where(np.diff(np.sign(diff)) != 0)[0]
    if len(sign_changes) == 0:
        raise ValueError("Binder cumulant curves for the two largest L do not cross on this grid")
    i = sign_changes[0]
    t0, t1 = T[i], T[i + 1]
    d0, d1 = diff[i], diff[i + 1]
    return float(t0 - d0 * (t1 - t0) / (d1 - d0))
