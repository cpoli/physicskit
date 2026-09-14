r"""Cross-sections and Mandelstam kinematics.

Uses the same natural-unit convention (:math:`\hbar=c=1`) as
:mod:`physicskit.particle.kinematics`.

- :func:`mandelstam_s`, :func:`mandelstam_t`, :func:`mandelstam_u` -- the
  three relativistic invariants of a 2-to-2 scattering process.
- :func:`rutherford_dsigma_domega`, :func:`impact_parameter` -- classical
  Rutherford scattering off a Coulomb potential.
"""

from __future__ import annotations

import numpy as np

__all__ = ["ALPHA_FS", "mandelstam_s", "mandelstam_t", "mandelstam_u", "rutherford_dsigma_domega", "impact_parameter"]

ALPHA_FS = 1 / 137.035999084
"""The fine-structure constant, :math:`\\alpha\\approx1/137.036`."""


def _dot_squared(p1, p2, sign):
    dE = p1.E + sign * p2.E
    dpx = p1.px + sign * p2.px
    dpy = p1.py + sign * p2.py
    dpz = p1.pz + sign * p2.pz
    return dE**2 - (dpx**2 + dpy**2 + dpz**2)


def mandelstam_s(p1, p2):
    r"""Mandelstam :math:`s=(p_1+p_2)^2`, the total invariant mass squared.

    Parameters
    ----------
    p1, p2 : physicskit.particle.kinematics.FourVector
        The two incoming (or, by the same formula, any two summed) particles.

    Returns
    -------
    float

    Examples
    --------
    >>> from physicskit.particle.kinematics import FourVector
    >>> p1 = FourVector(1.0, 0.0, 0.0, 0.6)
    >>> p2 = FourVector(1.0, 0.0, 0.0, -0.6)
    >>> round(mandelstam_s(p1, p2), 6)
    4.0
    """
    return float(_dot_squared(p1, p2, +1))


def mandelstam_t(p1, p3):
    r"""Mandelstam :math:`t=(p_1-p_3)^2`, the momentum-transfer invariant.

    Parameters
    ----------
    p1 : physicskit.particle.kinematics.FourVector
        An incoming particle.
    p3 : physicskit.particle.kinematics.FourVector
        The outgoing particle it is compared to.

    Returns
    -------
    float
    """
    return float(_dot_squared(p1, p3, -1))


def mandelstam_u(p1, p4):
    r"""Mandelstam :math:`u=(p_1-p_4)^2`, the other momentum-transfer invariant.

    Parameters
    ----------
    p1 : physicskit.particle.kinematics.FourVector
        An incoming particle.
    p4 : physicskit.particle.kinematics.FourVector
        The other outgoing particle.

    Returns
    -------
    float

    Notes
    -----
    For a 2-to-2 process :math:`1+2\to3+4`, :math:`s+t+u=\sum_i m_i^2`.
    """
    return float(_dot_squared(p1, p4, -1))


def rutherford_dsigma_domega(theta, Z1, Z2, E_kin, alpha=ALPHA_FS):
    r"""Rutherford differential cross section, in natural units.

    .. math::

        \frac{d\sigma}{d\Omega} = \left(\frac{Z_1Z_2\alpha}{4E_{\rm kin}}\right)^2
        \frac{1}{\sin^4(\theta/2)}

    Parameters
    ----------
    theta : array_like
        Scattering angle (radians). ``theta=0`` gives ``inf`` -- the
        genuine, integrable-but-divergent small-angle Rutherford behavior.
    Z1, Z2 : float
        Projectile and target charge numbers.
    E_kin : float
        Projectile kinetic energy.
    alpha : float, default=ALPHA_FS
        Coupling constant (the fine-structure constant, by default).

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> import numpy as np
    >>> theta = np.array([np.pi / 4, np.pi / 2])
    >>> vals = rutherford_dsigma_domega(theta, Z1=1, Z2=79, E_kin=5.0)
    >>> ratio = (vals[0] * np.sin(theta[0] / 2) ** 4) / (vals[1] * np.sin(theta[1] / 2) ** 4)
    >>> round(float(ratio), 6)
    1.0
    """
    theta = np.asarray(theta, dtype=float)
    prefactor = (Z1 * Z2 * alpha / (4.0 * E_kin)) ** 2
    with np.errstate(divide="ignore"):
        result = prefactor / np.sin(theta / 2.0) ** 4
    return result


def impact_parameter(theta, Z1, Z2, E_kin, alpha=ALPHA_FS):
    r"""Classical Rutherford impact parameter producing scattering angle ``theta``.

    .. math::

        b(\theta) = \frac{Z_1Z_2\alpha}{2E_{\rm kin}}\cot(\theta/2)

    Parameters
    ----------
    theta : array_like
        Scattering angle (radians).
    Z1, Z2 : float
        Projectile and target charge numbers.
    E_kin : float
        Projectile kinetic energy.
    alpha : float, default=ALPHA_FS
        Coupling constant.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(impact_parameter(np.pi, Z1=1, Z2=79, E_kin=5.0)), 6)
    0.0
    """
    theta = np.asarray(theta, dtype=float)
    return (Z1 * Z2 * alpha / (2.0 * E_kin)) / np.tan(theta / 2.0)
