r"""Quantum scars: eigenstate density concentrated on unstable classical periodic orbits.

Heller (1984) discovered that individual eigenstates of a classically
chaotic system are not always the featureless, ergodically-spread blobs
random-matrix intuition suggests: a small but robust fraction instead
show enhanced probability density in a tube around one particular
unstable classical periodic orbit -- a *scar* -- even though almost
every orbit nearby diverges from it exponentially fast. The bridge to
:mod:`physicskit.semiclassical.core.gutzwiller` is direct: the same
unstable periodic orbits that weight the Gutzwiller trace formula's
oscillating sum are exactly the orbits that scar individual eigenstates,
since eigenstates are (schematically) superpositions of trace-formula
terms.

:func:`bouncing_ball_energies` and :func:`bouncing_ball_orbit_points`
concern the single most famous scarred family in the Bunimovich stadium
billiard (:class:`physicskit.quantum.chapters.potentials.StadiumBilliard2D`):
the "bouncing ball" orbits bouncing straight up and down between the
flat top and bottom walls, which -- unlike a generic stadium orbit --
are only marginally unstable, so many low-lying eigenstates concentrate
along them. :func:`scar_enhancement` quantifies that concentration for
any eigenstate density, and :func:`husimi_projection_1d` builds a
non-periodic analog of :func:`physicskit.chaos.quantum.husimi.husimi_function`
for viewing a 1D wavefunction slice (e.g. along the billiard's flat
wall) in phase space.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "bouncing_ball_energies",
    "bouncing_ball_orbit_points",
    "scar_enhancement",
    "husimi_projection_1d",
]


def bouncing_ball_energies(R: float, hbar: float = 1.0, m: float = 1.0, n_max: int = 6) -> np.ndarray:
    r"""Predicted energies of the "bouncing ball" orbit family in a stadium billiard of cap radius ``R``.

    A trajectory launched perpendicular to the flat top and bottom walls
    of a Bunimovich stadium (see
    :class:`physicskit.quantum.chapters.potentials.StadiumBilliard2D`)
    bounces straight up and down forever, blind to the length ``L`` of
    the central rectangle -- exactly the motion of a particle in a 1D
    infinite square well of width :math:`2R`. Quantizing that 1D motion
    gives a leading-order prediction for where "bouncing ball" states
    (eigenstates concentrated on this orbit family) appear in the full
    2D spectrum:

    .. math::

       E_n = \frac{(n\pi\hbar)^2}{2m(2R)^2}, \qquad n=1,2,3,\dots

    Parameters
    ----------
    R : float
        Radius of the stadium's semicircular end-caps (half-height of
        the billiard).
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    m : float, default=1.0
        Particle mass.
    n_max : int, default=6
        Number of levels to compute.

    Returns
    -------
    ndarray, shape (n_max,)
        Predicted bouncing-ball energies, ascending.

    See Also
    --------
    bouncing_ball_orbit_points : The classical orbit these energies quantize.

    Examples
    --------
    This is exactly the ordinary infinite-square-well spectrum for a
    well of width :math:`2R`:

    >>> import numpy as np
    >>> R = 0.5
    >>> energies = bouncing_ball_energies(R, n_max=3)
    >>> exact = (np.arange(1, 4) * np.pi) ** 2 / (2 * (2 * R) ** 2)
    >>> bool(np.allclose(energies, exact))
    True
    """
    n = np.arange(1, n_max + 1)
    return (n * np.pi * hbar) ** 2 / (2.0 * m * (2.0 * R) ** 2)


def bouncing_ball_orbit_points(x0: float, R: float, n_bounces: int = 4) -> tuple:
    r"""Trace the classical "bouncing ball" orbit at horizontal position ``x0`` in a stadium billiard.

    A perpendicular launch from the bottom wall at :math:`(x_0,-R)`
    reflects straight back and forth off the flat top (:math:`y=R`) and
    bottom (:math:`y=-R`) walls, tracing the same vertical segment
    forever -- returned here as a zig-zag list of points suitable for
    overlaying on a density plot.

    Parameters
    ----------
    x0 : float
        Horizontal position of the orbit, with :math:`|x_0| < L/2`
        (strictly inside the flat central section of the stadium).
    R : float
        Radius of the stadium's semicircular end-caps.
    n_bounces : int, default=4
        Number of full up-down traversals to trace.

    Returns
    -------
    x, y : ndarray
        Coordinates tracing the orbit, shape ``(2 * n_bounces + 1,)``.

    See Also
    --------
    bouncing_ball_energies : Quantized energies of this orbit family.

    Examples
    --------
    >>> x, y = bouncing_ball_orbit_points(x0=0.2, R=0.5, n_bounces=2)
    >>> x
    array([0.2, 0.2, 0.2, 0.2, 0.2])
    >>> y
    array([-0.5,  0.5, -0.5,  0.5, -0.5])
    """
    y = np.array([-R if i % 2 == 0 else R for i in range(2 * n_bounces + 1)])
    x = np.full_like(y, x0)
    return x, y


def scar_enhancement(density: np.ndarray, X: np.ndarray, Y: np.ndarray, mask: np.ndarray, x0: float, half_width: float) -> float:
    r"""Density enhancement in a tube around a vertical bouncing-ball orbit, relative to the billiard average.

    .. math::

       \eta = \frac{\langle|\psi|^2\rangle_{\text{tube}}}{\langle|\psi|^2\rangle_{\text{billiard}}},
       \qquad \text{tube} = \{(x,y)\in\text{billiard} : |x-x_0|\le w\},

    a simple, direct measure of scarring: :math:`\eta\approx1` for an
    ergodically-spread (unscarred) state, since the tube then just
    samples a representative fraction of the whole density, while
    :math:`\eta\gg1` signals density piled up specifically along the
    orbit at :math:`x_0`.

    Parameters
    ----------
    density : ndarray
        Probability density :math:`|\psi(x,y)|^2` on a grid.
    X, Y : ndarray
        Coordinate meshgrid matching ``density``.
    mask : ndarray of bool
        ``True`` inside the billiard (e.g. from
        :meth:`physicskit.quantum.chapters.potentials.StadiumBilliard2D.mask`).
    x0 : float
        Horizontal position of the bouncing-ball orbit to test.
    half_width : float
        Half-width of the tube around ``x0``.

    Returns
    -------
    float
        The enhancement factor :math:`\eta`.

    Examples
    --------
    A density sharply concentrated in a narrow ridge at ``x0`` is
    strongly enhanced inside a tube covering that ridge:

    >>> import numpy as np
    >>> x = np.linspace(-2, 2, 400)
    >>> y = np.linspace(-1, 1, 200)
    >>> X, Y = np.meshgrid(x, y, indexing="ij")
    >>> mask = np.ones_like(X, dtype=bool)
    >>> density = np.exp(-(X - 0.3) ** 2 / (2 * 0.05 ** 2))
    >>> eta = scar_enhancement(density, X, Y, mask, x0=0.3, half_width=0.15)
    >>> bool(12.0 < eta < 14.0)
    True

    An (exactly) uniform density is not enhanced anywhere:

    >>> uniform = np.ones_like(X)
    >>> round(scar_enhancement(uniform, X, Y, mask, x0=0.3, half_width=0.15), 8)
    1.0
    """
    tube = mask & (np.abs(X - x0) <= half_width)
    return float(np.mean(density[tube]) / np.mean(density[mask]))


def husimi_projection_1d(
    psi: np.ndarray,
    s: np.ndarray,
    hbar: float = 1.0,
    sigma: float | None = None,
    resolution: int = 60,
    s0_range: tuple | None = None,
    p0_range: tuple | None = None,
) -> tuple:
    r"""Husimi (coherent-state) phase-space projection of a 1D wavefunction slice on an open interval.

    The non-periodic counterpart of
    :func:`physicskit.chaos.quantum.husimi.husimi_function`: overlaps
    ``psi`` with ordinary (non-periodized) coherent states
    :math:`g_{s_0,p_0}(s)=\exp[-(s-s_0)^2/2\sigma^2+ip_0(s-s_0)/\hbar]`
    at every point of an :math:`(s_0,p_0)` grid, suited to a slice taken
    along an open boundary -- e.g. the flat top wall of a stadium
    billiard, where :func:`bouncing_ball_orbit_points` lives -- rather
    than a periodic domain.

    Parameters
    ----------
    psi : ndarray of complex
        Wavefunction values sampled on ``s``.
    s : ndarray
        Uniform 1D coordinate grid ``psi`` is sampled on.
    hbar : float, default=1.0
        Value of :math:`\hbar` to use.
    sigma : float, optional
        Coherent-state position width. Defaults to :math:`\sqrt{\hbar}`
        (the minimum-uncertainty, equal-spread-in-natural-units choice).
    resolution : int, default=60
        Number of grid points along each of the :math:`s_0`, :math:`p_0` axes.
    s0_range : (float, float), optional
        Range of :math:`s_0` to scan. Defaults to ``(s.min(), s.max())``.
    p0_range : (float, float), optional
        Range of :math:`p_0` to scan. Defaults to
        :math:`\pm\pi\hbar/\Delta s` (the grid's Nyquist momentum).

    Returns
    -------
    S0, P0 : ndarray, shape (resolution, resolution)
        Phase-space grid (``indexing="ij"``).
    husimi : ndarray, shape (resolution, resolution)
        The Husimi distribution, normalized to a peak value of 1.

    See Also
    --------
    scar_enhancement : A simpler, single-number scarring diagnostic in
        position space rather than phase space.

    Examples
    --------
    A Gaussian wave packet's Husimi projection peaks at its own
    position and momentum:

    >>> import numpy as np
    >>> s = np.linspace(-10, 10, 2000)
    >>> s0_true, p0_true, w = 2.0, 3.0, 1.0
    >>> psi = np.exp(-(s - s0_true) ** 2 / (2 * w ** 2)) * np.exp(1j * p0_true * s / 1.0)
    >>> S0, P0, H = husimi_projection_1d(psi, s, hbar=1.0, sigma=w, resolution=80, s0_range=(-2, 6), p0_range=(-2, 8))
    >>> i, j = np.unravel_index(np.argmax(H), H.shape)
    >>> bool(abs(S0[i, j] - s0_true) < 0.2 and abs(P0[i, j] - p0_true) < 0.2)
    True
    """
    s = np.asarray(s, dtype=float)
    ds = s[1] - s[0]
    if sigma is None:
        sigma = np.sqrt(hbar)
    if s0_range is None:
        s0_range = (float(s.min()), float(s.max()))
    if p0_range is None:
        p_max = np.pi * hbar / ds
        p0_range = (-p_max, p_max)

    s0 = np.linspace(s0_range[0], s0_range[1], resolution)
    p0 = np.linspace(p0_range[0], p0_range[1], resolution)
    S0, P0 = np.meshgrid(s0, p0, indexing="ij")

    ds_grid = s[None, None, :] - S0[:, :, None]
    coherent = np.exp(-(ds_grid**2) / (2.0 * sigma**2) + 1j * P0[:, :, None] * ds_grid / hbar)
    overlap = np.tensordot(np.conj(coherent), psi, axes=([2], [0])) * ds

    husimi = np.abs(overlap) ** 2
    husimi /= husimi.max()
    return S0, P0, husimi
