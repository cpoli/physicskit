r"""Ginzburg-Landau theory: the 1950 phenomenological theory of continuous phase transitions.

Landau and Ginzburg proposed that a superconductor (or any system undergoing
a continuous symmetry-breaking transition) is described entirely by a
complex order parameter :math:`\psi(\mathbf{r})` -- here, the Cooper-pair
condensate wavefunction -- through a free-energy functional expanded in
powers of :math:`\psi` and its gradient,

.. math::

   f[\psi] = a|\psi|^2 + \frac{b}{2}|\psi|^4
   + \frac{\hbar^2}{2m}|\nabla\psi|^2,

built from symmetry alone, with no reference to the microscopic pairing
mechanism BCS would supply seven years later. Minimizing :math:`f` gives a
uniform condensate density :math:`|\psi_0|^2 = -a/b` below the transition
(:math:`a<0`) and two emergent length scales -- the coherence length
:math:`\xi` over which :math:`\psi` heals back from a boundary, and the
magnetic penetration depth :math:`\lambda` -- whose ratio
:math:`\kappa=\lambda/\xi` (the Ginzburg-Landau parameter) alone decides
whether a superconductor is Type I or Type II.

Uses the same natural-unit convention (:math:`\hbar=e=1`, and here also
:math:`\mu_0=1`) as the rest of :mod:`physicskit.condensed`.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "gl_free_energy_density",
    "gl_equilibrium_order_parameter",
    "gl_coherence_length",
    "gl_penetration_depth",
    "ginzburg_landau_parameter",
    "gl_order_parameter_profile",
]


def gl_free_energy_density(psi, a: float, b: float, grad_psi=0.0, hbar: float = 1.0, m: float = 1.0):
    r"""Ginzburg-Landau free energy density :math:`f = a|\psi|^2 + \tfrac{b}{2}|\psi|^4 + \tfrac{\hbar^2}{2m}|\nabla\psi|^2`.

    Parameters
    ----------
    psi : complex or array_like
        Order parameter value(s).
    a : float
        Quadratic coefficient. Changes sign at the transition (``a < 0``
        in the ordered phase, ``a > 0`` in the disordered phase).
    b : float
        Quartic coefficient, ``b > 0`` for stability.
    grad_psi : complex or array_like, default=0.0
        Gradient :math:`\nabla\psi`, same shape as ``psi``.
    hbar : float, default=1.0
        Reduced Planck constant.
    m : float, default=1.0
        Effective mass of the condensate.

    Returns
    -------
    float or ndarray

    Examples
    --------
    >>> gl_free_energy_density(psi=0.0, a=-1.0, b=1.0)
    0.0
    >>> psi0 = gl_equilibrium_order_parameter(a=-1.0, b=1.0)
    >>> round(gl_free_energy_density(psi0, a=-1.0, b=1.0), 6)
    -0.5
    """
    psi = np.asarray(psi, dtype=complex)
    grad_psi = np.asarray(grad_psi, dtype=complex)
    f = a * np.abs(psi) ** 2 + 0.5 * b * np.abs(psi) ** 4 + (hbar**2 / (2 * m)) * np.abs(grad_psi) ** 2
    return float(f) if f.ndim == 0 else f


def gl_equilibrium_order_parameter(a: float, b: float) -> float:
    r"""Equilibrium (uniform, field-free) order parameter magnitude :math:`|\psi_0|=\sqrt{-a/b}`.

    Parameters
    ----------
    a : float
        Quadratic coefficient.
    b : float
        Quartic coefficient, ``b > 0``.

    Returns
    -------
    float
        :math:`\sqrt{-a/b}` if ``a < 0`` (ordered phase), else ``0.0``
        (disordered phase, only the normal state minimizes :math:`f`).

    Examples
    --------
    >>> gl_equilibrium_order_parameter(a=-2.0, b=2.0)
    1.0
    >>> gl_equilibrium_order_parameter(a=1.0, b=2.0)
    0.0
    """
    return float(np.sqrt(-a / b)) if a < 0 else 0.0


def gl_coherence_length(a: float, hbar: float = 1.0, m: float = 1.0) -> float:
    r"""Ginzburg-Landau coherence length :math:`\xi = \hbar/\sqrt{2m|a|}`.

    The length scale over which the order parameter heals back to its bulk
    value after being suppressed at a boundary or a vortex core.

    Parameters
    ----------
    a : float
        Quadratic coefficient (only its magnitude matters).
    hbar : float, default=1.0
        Reduced Planck constant.
    m : float, default=1.0
        Effective mass of the condensate.

    Returns
    -------
    float

    Examples
    --------
    >>> gl_coherence_length(a=-0.5)
    1.0
    """
    return float(hbar / np.sqrt(2 * m * abs(a)))


def gl_penetration_depth(psi0: float, e: float = 1.0, m: float = 1.0) -> float:
    r"""London penetration depth :math:`\lambda = \sqrt{m/(e^2|\psi_0|^2)}`.

    The length scale over which an external magnetic field is screened
    from a superconductor's interior by the supercurrent it induces in the
    condensate of density :math:`|\psi_0|^2`.

    Parameters
    ----------
    psi0 : float
        Equilibrium order parameter magnitude (see
        :func:`gl_equilibrium_order_parameter`).
    e : float, default=1.0
        Cooper-pair charge magnitude.
    m : float, default=1.0
        Effective mass of the condensate.

    Returns
    -------
    float

    Examples
    --------
    >>> gl_penetration_depth(psi0=1.0)
    1.0
    """
    return float(np.sqrt(m / (e**2 * psi0**2)))


def ginzburg_landau_parameter(coherence_length: float, penetration_depth: float) -> float:
    r"""Ginzburg-Landau parameter :math:`\kappa = \lambda/\xi`.

    The single dimensionless number that decides a superconductor's
    response to a magnetic field: :math:`\kappa < 1/\sqrt2` is Type I
    (the normal-superconducting interface has positive surface energy, and
    the field is excluded entirely below :math:`H_c`); :math:`\kappa >
    1/\sqrt2` is Type II (negative surface energy favors flux penetrating
    as an Abrikosov vortex lattice between :math:`H_{c1}` and
    :math:`H_{c2}`).

    Parameters
    ----------
    coherence_length : float
        :math:`\xi`, from :func:`gl_coherence_length`.
    penetration_depth : float
        :math:`\lambda`, from :func:`gl_penetration_depth`.

    Returns
    -------
    float

    Examples
    --------
    >>> round(ginzburg_landau_parameter(coherence_length=1.0, penetration_depth=1.0), 4)
    1.0
    """
    return float(penetration_depth / coherence_length)


def gl_order_parameter_profile(x, xi: float) -> np.ndarray:
    r"""Order parameter healing profile :math:`\psi(x)/\psi_0 = \tanh(x/(\sqrt2\,\xi))`.

    The exact solution of the dimensionless Ginzburg-Landau equation
    :math:`\xi^2\psi'' = \psi^3-\psi` for a condensate pinned to zero at a
    boundary (``x = 0``, e.g. a normal-superconducting interface) and
    recovering its bulk value far from it -- the direct, textbook
    illustration of the coherence length as a *healing length*.

    Parameters
    ----------
    x : array_like
        Distance from the boundary.
    xi : float
        Coherence length, from :func:`gl_coherence_length`.

    Returns
    -------
    ndarray
        :math:`\psi(x)/\psi_0`, in :math:`[0, 1)` for :math:`x \geq 0`.

    Examples
    --------
    >>> import numpy as np
    >>> round(float(gl_order_parameter_profile(0.0, xi=1.0)), 8)
    0.0
    >>> bool(gl_order_parameter_profile(np.array([10.0]), xi=1.0)[0] > 0.999)
    True
    """
    x = np.asarray(x, dtype=float)
    return np.tanh(x / (np.sqrt(2) * xi))
