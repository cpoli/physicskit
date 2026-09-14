r"""Gaussian beam propagation: the complex beam parameter, ABCD transformation, and higher-order modes.

A paraxial, monochromatic beam with a Gaussian transverse intensity profile
is completely described, at any position :math:`z` along its propagation
axis, by a single complex number -- the beam parameter :math:`q(z)`,
introduced by Kogelnik & Li (*Appl. Opt.* **5**, 1550, 1966) -- through

.. math::

    \frac{1}{q(z)} = \frac{1}{R(z)} - i\,\frac{\lambda}{\pi w(z)^2},

where :math:`w(z)` is the :math:`1/e^2` intensity radius and :math:`R(z)`
the radius of curvature of the wavefronts. Propagation through any paraxial
optical system described by a ray-transfer (ABCD) matrix updates :math:`q`
by the same bilinear (Mobius) transformation that acts on ray-tracing
matrices, which is what makes the complex-:math:`q` formalism so useful:
free-space propagation, lenses, mirrors, and interfaces all become 2x2
matrix multiplications, exactly as in ordinary ray optics. This module also
covers the free-space eigenmodes of the paraxial wave equation built on top
of the fundamental Gaussian -- the Hermite-Gaussian and Laguerre-Gaussian
mode families -- and the empirical :math:`M^2` beam-quality factor used to
describe real, non-diffraction-limited laser beams.

The ABCD matrix convention used throughout (shared with
:mod:`physicskit.optics.ray`) is ``M = [[A, B], [C, D]]`` acting on a ray
state ``[y, theta]`` as ``state_out = M @ state_in``; e.g. free space of
length ``d`` is ``[[1, d], [0, 1]]`` and a thin lens of focal length ``f``
is ``[[1, 0], [-1/f, 1]]``.
"""

from __future__ import annotations

import numpy as np
from scipy.special import eval_genlaguerre, eval_hermite

__all__ = [
    "propagate_q",
    "q_to_beam_params",
    "GaussianBeam",
    "hermite_gaussian_mode",
    "laguerre_gaussian_mode",
    "m2_beam_waist",
]


def propagate_q(q, M):
    r"""Propagate a complex beam parameter through a paraxial optical system.

    Applies the same bilinear transformation used for ray-transfer (ABCD)
    matrices,

    .. math::

        q_{\text{out}} = \frac{A q + B}{C q + D},

    with :math:`A, B, C, D` the entries of ``M`` in the convention
    ``M = [[A, B], [C, D]]``.

    Parameters
    ----------
    q : complex
        Beam parameter before the system.
    M : ndarray of shape (2, 2)
        Ray-transfer (ABCD) matrix of the optical system.

    Returns
    -------
    complex
        Beam parameter ``q_out`` after the system.

    Examples
    --------
    >>> import numpy as np
    >>> free_space = np.array([[1.0, 2.0], [0.0, 1.0]])
    >>> propagate_q(1j, free_space)
    (2+1j)
    """
    M = np.asarray(M)
    A, B, C, D = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    q = complex(q)
    return complex((A * q + B) / (C * q + D))


def q_to_beam_params(q, wavelength):
    r"""Extract the beam radius and radius of curvature from a complex beam parameter.

    Inverts

    .. math::

        \frac{1}{q} = \frac{1}{R} - i\,\frac{\lambda}{\pi w^2}.

    At the beam waist :math:`\mathrm{Re}(1/q) = 0`, so ``R`` would formally
    diverge; this is handled explicitly and ``np.inf`` is returned there
    instead of raising or silently producing ``nan``.

    Parameters
    ----------
    q : complex
        Complex beam parameter.
    wavelength : float
        Wavelength (in the same length units as ``q``).

    Returns
    -------
    w : float
        Beam radius (:math:`1/e^2` intensity radius).
    R : float
        Radius of curvature of the wavefront; ``np.inf`` at the waist.

    Examples
    --------
    >>> import numpy as np
    >>> zR = 3.0
    >>> w, R = q_to_beam_params(1j * zR, wavelength=0.5e-3)
    >>> round(float(w), 6), R
    (0.021851, inf)
    """
    q = complex(q)
    inv_q = 1.0 / q
    re, im = inv_q.real, inv_q.imag
    mag = abs(inv_q)
    if mag == 0.0 or abs(re) < 1e-10 * mag:
        R = np.inf
    else:
        R = 1.0 / re
    w = np.sqrt(-wavelength / (np.pi * im))
    return w, R


class GaussianBeam:
    r"""A fundamental (:math:`\mathrm{TEM}_{00}`) Gaussian laser beam.

    Parameters
    ----------
    wavelength : float
        Wavelength, in the same length units as ``w0`` and ``z0``.
    w0 : float
        Waist radius (:math:`1/e^2` intensity radius at the narrowest point).
    z0 : float, default=0.0
        Axial position of the waist.
    """

    def __init__(self, wavelength, w0, z0=0.0):
        self.wavelength = wavelength
        self.w0 = w0
        self.z0 = z0

    @property
    def rayleigh_range(self):
        r"""Rayleigh range, :math:`z_R = \pi w_0^2/\lambda`."""
        return np.pi * self.w0**2 / self.wavelength

    def waist(self, z):
        r"""Beam radius at axial position ``z``.

        .. math::

            w(z) = w_0 \sqrt{1 + \left(\frac{z - z_0}{z_R}\right)^2}

        Parameters
        ----------
        z : float or array_like
            Axial position(s).

        Returns
        -------
        float or ndarray
        """
        zR = self.rayleigh_range
        return self.w0 * np.sqrt(1.0 + ((np.asarray(z, dtype=float) - self.z0) / zR) ** 2)

    def radius_of_curvature(self, z):
        r"""Wavefront radius of curvature at axial position ``z``.

        .. math::

            R(z) = (z - z_0)\left[1 + \left(\frac{z_R}{z - z_0}\right)^2\right]

        with :math:`R(z_0) = \infty` (the wavefront is flat at the waist),
        handled explicitly rather than relying on the indeterminate-form
        arithmetic of the formula above.

        Parameters
        ----------
        z : float or array_like
            Axial position(s).

        Returns
        -------
        float or ndarray
        """
        zR = self.rayleigh_range
        dz = np.asarray(z, dtype=float) - self.z0
        dz_safe = np.where(dz == 0.0, 1.0, dz)
        R = np.where(dz == 0.0, np.inf, dz_safe + zR**2 / dz_safe)
        if np.isscalar(z) or np.ndim(z) == 0:
            return float(R)
        return R

    def gouy_phase(self, z):
        r"""Gouy phase at axial position ``z``, :math:`\zeta(z) = \arctan[(z-z_0)/z_R]`.

        Parameters
        ----------
        z : float or array_like
            Axial position(s).

        Returns
        -------
        float or ndarray
        """
        zR = self.rayleigh_range
        return np.arctan((np.asarray(z, dtype=float) - self.z0) / zR)

    def q_parameter(self, z):
        r"""Complex beam parameter at axial position ``z``, :math:`q(z) = (z-z_0) + i z_R`.

        Parameters
        ----------
        z : float
            Axial position.

        Returns
        -------
        complex
        """
        zR = self.rayleigh_range
        return complex(z - self.z0, zR)

    @property
    def divergence_angle(self):
        r"""Far-field half-angle divergence, :math:`\theta = \lambda/(\pi w_0)`."""
        return self.wavelength / (np.pi * self.w0)


def hermite_gaussian_mode(x, y, z, beam, m, n):
    r"""Hermite-Gaussian :math:`\mathrm{TEM}_{mn}` mode amplitude.

    .. math::

        u_{mn}(x, y, z) = \frac{w_0}{w(z)}\,
            H_m\!\left(\frac{\sqrt2\,x}{w(z)}\right)
            H_n\!\left(\frac{\sqrt2\,y}{w(z)}\right)
            \exp\!\left[-\frac{x^2+y^2}{w(z)^2}\right]
            \exp\!\left[-\frac{ik(x^2+y^2)}{2R(z)}\right]
            \exp\!\left[i(m+n+1)\zeta(z)\right] \exp(-ikz)

    with :math:`k = 2\pi/\lambda`. :math:`m=n=0` reduces to the fundamental
    Gaussian mode carried by ``beam``.

    Parameters
    ----------
    x, y : array_like
        Transverse coordinates (broadcastable).
    z : float
        Axial position.
    beam : GaussianBeam
        The underlying fundamental-mode beam (sets :math:`w_0`, :math:`z_0`,
        :math:`\lambda`).
    m, n : int
        Transverse mode indices along :math:`x` and :math:`y`.

    Returns
    -------
    ndarray
        Complex field amplitude, broadcast shape of ``x`` and ``y``.

    Examples
    --------
    >>> import numpy as np
    >>> beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    >>> u00 = hermite_gaussian_mode(0.0, 0.0, 0.0, beam, 0, 0)
    >>> round(float(abs(u00)), 6)
    1.0
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    k = 2.0 * np.pi / beam.wavelength
    w = beam.waist(z)
    R = beam.radius_of_curvature(z)
    zeta = beam.gouy_phase(z)
    r2 = x**2 + y**2
    curvature_phase = 0.0 if np.isinf(R) else k * r2 / (2.0 * R)
    amplitude = (beam.w0 / w) * eval_hermite(m, np.sqrt(2.0) * x / w) * eval_hermite(n, np.sqrt(2.0) * y / w) * np.exp(-r2 / w**2)
    phase = np.exp(-1j * curvature_phase) * np.exp(1j * (m + n + 1) * zeta) * np.exp(-1j * k * z)
    return (amplitude * phase).astype(complex)


def laguerre_gaussian_mode(r, phi, z, beam, l, p):
    r"""Laguerre-Gaussian :math:`\mathrm{LG}_p^l` mode amplitude.

    .. math::

        u_{lp}(r, \phi, z) = \frac{w_0}{w(z)}
            \left(\frac{r\sqrt2}{w(z)}\right)^{|l|}
            L_p^{|l|}\!\left(\frac{2r^2}{w(z)^2}\right)
            \exp\!\left[-\frac{r^2}{w(z)^2}\right]
            \exp\!\left[-\frac{ikr^2}{2R(z)}\right]
            \exp(il\phi)
            \exp\!\left[i(|l|+2p+1)\zeta(z)\right] \exp(-ikz)

    with :math:`k = 2\pi/\lambda` and :math:`L_p^{|l|}` the associated
    Laguerre polynomial.

    Parameters
    ----------
    r, phi : array_like
        Polar transverse coordinates (broadcastable); ``r`` is the radial
        distance from the axis, ``phi`` the azimuthal angle.
    z : float
        Axial position.
    beam : GaussianBeam
        The underlying fundamental-mode beam.
    l : int
        Azimuthal (orbital) mode index.
    p : int
        Radial mode index.

    Returns
    -------
    ndarray
        Complex field amplitude, broadcast shape of ``r`` and ``phi``.

    Examples
    --------
    >>> import numpy as np
    >>> beam = GaussianBeam(wavelength=1.0, w0=1.0, z0=0.0)
    >>> u00 = laguerre_gaussian_mode(0.0, 0.0, 0.0, beam, 0, 0)
    >>> round(float(abs(u00)), 6)
    1.0
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    k = 2.0 * np.pi / beam.wavelength
    w = beam.waist(z)
    R = beam.radius_of_curvature(z)
    zeta = beam.gouy_phase(z)
    curvature_phase = 0.0 if np.isinf(R) else k * r**2 / (2.0 * R)
    amplitude = (beam.w0 / w) * (r * np.sqrt(2.0) / w) ** abs(l) * eval_genlaguerre(p, abs(l), 2.0 * r**2 / w**2) * np.exp(-(r**2) / w**2)
    phase = np.exp(-1j * curvature_phase) * np.exp(1j * l * phi) * np.exp(1j * (abs(l) + 2 * p + 1) * zeta) * np.exp(-1j * k * z)
    return (amplitude * phase).astype(complex)


def m2_beam_waist(z, wavelength, w0, M2=1.0):
    r"""Beam radius of a non-ideal (:math:`M^2 > 1`) real laser beam, waist fixed at :math:`z=0`.

    .. math::

        w(z) = w_0 \sqrt{1 + \left(\frac{z}{z_{R,\text{eff}}}\right)^2},
        \qquad z_{R,\text{eff}} = \frac{\pi w_0^2}{M^2 \lambda}

    A beam-quality factor :math:`M^2 \ge 1` (equal to 1 for an ideal
    diffraction-limited Gaussian beam) reduces the effective Rayleigh range,
    so a real beam of the same waist diverges faster than the ideal case.

    Parameters
    ----------
    z : float or array_like
        Axial position(s), measured from the waist at :math:`z=0`.
    wavelength : float
        Wavelength.
    w0 : float
        Waist radius.
    M2 : float, default=1.0
        Beam-quality factor, :math:`M^2 \ge 1`.

    Returns
    -------
    float or ndarray
        Beam radius ``w(z)``.

    Examples
    --------
    >>> round(float(m2_beam_waist(0.0, wavelength=0.5e-3, w0=0.1, M2=2.0)), 6)
    0.1
    """
    zR_eff = np.pi * w0**2 / (M2 * wavelength)
    return w0 * np.sqrt(1.0 + (np.asarray(z, dtype=float) / zR_eff) ** 2)
