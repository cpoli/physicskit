r"""Two-flavor vacuum neutrino oscillations.

- :func:`oscillation_probability` -- the standard two-flavor
  :math:`\nu_e\to\nu_\mu` vacuum oscillation probability.
- :func:`survival_probability` -- its complement, :math:`\nu_e\to\nu_e`.

**Simplification**: a full treatment has three flavors, a PMNS matrix
with three mixing angles and a CP-violating phase, and matter (MSW)
effects for neutrinos propagating through the Earth or the Sun. The
two-flavor vacuum-oscillation formula used here is the standard
leading-order approximation whenever a single mass-squared splitting
dominates (as :math:`\Delta m^2_{21}` or :math:`\Delta m^2_{31}}` each
individually do in different regimes), and is the textbook starting
point for any oscillation demonstration (see the Particle Data Group's
"Neutrino Mixing" review).
"""

from __future__ import annotations

import numpy as np

__all__ = ["oscillation_probability", "survival_probability"]


def oscillation_probability(L, E, theta, delta_m2):
    r"""Two-flavor vacuum oscillation probability :math:`P(\nu_e\to\nu_\mu)`.

    .. math::

        P(\nu_e\to\nu_\mu) = \sin^2(2\theta)\,
        \sin^2\!\left(1.267\,\frac{\Delta m^2 L}{E}\right)

    the standard PDG form in practical units: :math:`L` in km, :math:`E`
    in GeV, :math:`\Delta m^2` in :math:`{\rm eV}^2` (the numerical
    constant 1.267 absorbs :math:`\hbar,c` and the unit conversion; see
    the Particle Data Group's "Neutrino Mixing" review).

    Parameters
    ----------
    L : array_like
        Baseline (distance traveled), in km.
    E : array_like
        Neutrino energy, in GeV.
    theta : float
        Mixing angle, in radians.
    delta_m2 : float
        Mass-squared splitting, in :math:`{\rm eV}^2`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(oscillation_probability(0.0, 1.0, 0.5, 2.5e-3)), 10)
    0.0
    >>> theta = np.pi / 4  # maximal mixing
    >>> L_E = np.pi / 2 / (1.267 * 2.5e-3)  # argument = pi/2 -> sin^2 = 1
    >>> round(float(oscillation_probability(L_E, 1.0, theta, 2.5e-3)), 6)
    1.0
    """
    L = np.asarray(L, dtype=float)
    E = np.asarray(E, dtype=float)
    return np.sin(2.0 * theta) ** 2 * np.sin(1.267 * delta_m2 * L / E) ** 2


def survival_probability(L, E, theta, delta_m2):
    r"""Two-flavor vacuum survival probability, :math:`P(\nu_e\to\nu_e)=1-P(\nu_e\to\nu_\mu)`.

    Parameters
    ----------
    L, E, theta, delta_m2
        See :func:`oscillation_probability`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(survival_probability(0.0, 1.0, 0.5, 2.5e-3)), 10)
    1.0
    """
    return 1.0 - oscillation_probability(L, E, theta, delta_m2)
