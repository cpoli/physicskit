r"""The Navarro-Frenk-White dark matter halo profile and rotation curves.

Uses **gravitational units with** :math:`G=1` by default -- every function
accepts ``G`` as a keyword argument, the same convention
:mod:`physicskit.relativity` uses for its own geometrized units.

- :func:`nfw_density`, :func:`nfw_enclosed_mass`, :func:`nfw_potential`
  -- the NFW halo profile and its enclosed mass and potential.
- :func:`circular_velocity` -- the rotation curve produced by any
  enclosed-mass profile.
"""

from __future__ import annotations

import numpy as np

__all__ = ["nfw_density", "nfw_enclosed_mass", "nfw_potential", "circular_velocity"]


def nfw_density(r, rho_s, r_s):
    r"""The Navarro-Frenk-White density profile.

    .. math::

        \rho_{\rm NFW}(r) = \frac{\rho_s}{(r/r_s)(1+r/r_s)^2}

    Parameters
    ----------
    r : array_like
        Radius.
    rho_s : float
        Characteristic density.
    r_s : float
        Scale radius.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(nfw_density(1.0, rho_s=1.0, r_s=1.0)), 6)
    0.25
    """
    x = np.asarray(r, dtype=float) / r_s
    return rho_s / (x * (1.0 + x) ** 2)


def nfw_enclosed_mass(r, rho_s, r_s, G=1.0):
    r"""Mass enclosed within radius ``r`` for an NFW halo.

    .. math::

        M(<r) = 4\pi\rho_s r_s^3\left[\ln\!\left(1+\frac{r}{r_s}\right)
        - \frac{r/r_s}{1+r/r_s}\right]

    ``G`` is accepted for interface consistency with the rest of this
    module but is not used by this formula.

    Parameters
    ----------
    r : array_like
        Radius.
    rho_s : float
        Characteristic density.
    r_s : float
        Scale radius.
    G : float, default=1.0
        Unused; accepted for a consistent function signature.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(nfw_enclosed_mass(1.0, rho_s=1.0, r_s=1.0)), 6)
    2.427159
    """
    del G
    x = np.asarray(r, dtype=float) / r_s
    return 4.0 * np.pi * rho_s * r_s**3 * (np.log(1.0 + x) - x / (1.0 + x))


def nfw_potential(r, rho_s, r_s, G=1.0):
    r"""The NFW gravitational potential.

    .. math::

        \Phi(r) = -\frac{4\pi G\rho_sr_s^3}{r}\ln\!\left(1+\frac{r}{r_s}\right)

    Parameters
    ----------
    r : array_like
        Radius.
    rho_s : float
        Characteristic density.
    r_s : float
        Scale radius.
    G : float, default=1.0
        Gravitational constant.

    Returns
    -------
    ndarray or float
    """
    r = np.asarray(r, dtype=float)
    x = r / r_s
    return -4.0 * np.pi * G * rho_s * r_s**3 * np.log(1.0 + x) / r


def circular_velocity(r, mass_enclosed_func, G=1.0):
    r"""Circular orbital speed for any enclosed-mass profile, :math:`v_c(r)=\sqrt{GM(<r)/r}`.

    Parameters
    ----------
    r : array_like
        Radius.
    mass_enclosed_func : callable
        ``mass_enclosed_func(r) -> mass``, e.g.
        ``lambda r: nfw_enclosed_mass(r, rho_s, r_s)``.
    G : float, default=1.0
        Gravitational constant.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> M0 = 5.0
    >>> round(float(circular_velocity(2.0, lambda r: M0, G=1.0)), 6)
    1.581139
    """
    r = np.asarray(r, dtype=float)
    M = mass_enclosed_func(r)
    return np.sqrt(G * M / r)
