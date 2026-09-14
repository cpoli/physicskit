"""Point-source gravitational lensing: Einstein rings and multiple images.

When a point source, a point (or compact) lens, and an observer are exactly
aligned, the source's light is bent symmetrically around the lens into a
complete ring -- an Einstein ring, predicted by Einstein in 1936 (though he
believed it would never be observable) and first imaged for a galaxy-scale
lens in 1988. For any other alignment, a point lens splits a single source
into exactly two images, one on each side of the lens, with a combined
brightness that can be dramatically magnified -- the basis of gravitational
microlensing surveys used to detect exoplanets and dark, compact objects.
This module solves the thin-lens equation analytically in the weak-field
(point-mass) limit, and provides the exact (numerically integrated)
deflection angle for exploring strong-field corrections near the photon
sphere.
"""

from __future__ import annotations

import numpy as np

__all__ = ["PointMassLens", "exact_deflection_angle"]


class PointMassLens:
    """A point-mass (weak-field) gravitational lens: Einstein ring, images, and magnification.

    Uses the thin-lens approximation with the leading-order deflection
    angle :math:`\\hat\\alpha(b) = 4M/b`, valid whenever the source is not
    aligned so closely that its image angle approaches the photon sphere
    (see :func:`exact_deflection_angle` for the strong-field case).

    Parameters
    ----------
    M : float
        Lens mass, in geometrized units.
    D_L : float
        Observer-lens distance, in the same geometrized length units as ``M``.
    D_S : float
        Observer-source distance (:math:`D_S > D_L`), in the same units as ``M``.

    Notes
    -----
    Every formula here is a ratio of ``M``, ``D_L``, ``D_S`` (angles and
    magnifications, not an absolute strain), so it is scale-invariant: a
    normalized convention (``M=1``, distances "in units of M") gives
    exactly the same angles as real SI-consistent geometrized values would.
    Unlike :class:`~physicskit.relativity.chapters.gw_merger.BinaryMerger`,
    there is no absolute-scale trap here -- any self-consistent unit choice
    works.

    Attributes
    ----------
    D_LS : float
        Lens-source distance, :math:`D_S - D_L` (flat-space subtraction;
        this is a local, non-cosmological thin-lens treatment).
    """

    def __init__(self, M, D_L, D_S):
        if D_S <= D_L:
            raise ValueError("D_S must exceed D_L (the source must be behind the lens)")
        self.M = M
        self.D_L = D_L
        self.D_S = D_S
        self.D_LS = D_S - D_L

    def einstein_angle(self):
        """Einstein ring angular radius, :math:`\\theta_E = \\sqrt{4M D_{LS}/(D_L D_S)}`.

        The angular radius at which a perfectly aligned (:math:`\\beta=0`)
        point source appears as a complete ring.

        Returns
        -------
        float
        """
        return np.sqrt(4.0 * self.M * self.D_LS / (self.D_L * self.D_S))

    def image_angles(self, beta):
        """Angular positions of the two images of a source at angular offset ``beta``.

        Solves the thin-lens equation
        :math:`\\beta = \\theta - \\theta_E^2/\\theta` (equivalent to a
        quadratic in :math:`\\theta`) exactly:

        .. math::

            \\theta_\\pm = \\frac{\\beta \\pm \\sqrt{\\beta^2 + 4\\theta_E^2}}{2}

        Parameters
        ----------
        beta : float
            True (unlensed) angular offset of the source from the lens.

        Returns
        -------
        theta_plus : float
            Position of the primary image, on the same side as the source
            (:math:`\\theta_+ > 0`).
        theta_minus : float
            Position of the secondary image, on the opposite side
            (:math:`\\theta_- < 0`), closer to the lens and fainter.

        Examples
        --------
        >>> lens = PointMassLens(M=1.0, D_L=1000.0, D_S=2000.0)
        >>> theta_plus, theta_minus = lens.image_angles(beta=0.0)
        >>> bool(np.isclose(theta_plus, lens.einstein_angle()))
        True
        >>> bool(np.isclose(theta_minus, -lens.einstein_angle()))
        True
        """
        theta_E = self.einstein_angle()
        discriminant = np.sqrt(beta**2 + 4.0 * theta_E**2)
        theta_plus = (beta + discriminant) / 2.0
        theta_minus = (beta - discriminant) / 2.0
        return theta_plus, theta_minus

    def magnification(self, beta):
        """Magnification of each image, and their sum (total observed brightening).

        Standard point-lens (Paczynski 1986) result, with
        :math:`u = \\beta/\\theta_E`:

        .. math::

            \\mu_\\pm = \\frac{1}{2}\\left[\\frac{u^2+2}{u\\sqrt{u^2+4}} \\pm 1\\right]

        Parameters
        ----------
        beta : float
            True angular offset of the source from the lens (``beta != 0``).

        Returns
        -------
        mu_plus : float
            Magnification of the primary image (always :math:`> 1`).
        mu_minus : float
            Magnification of the secondary image (its absolute value is
            :math:`< 1`; it has opposite parity, conventionally signed
            negative).
        total : float
            Total observed magnification, :math:`|\\mu_+| + |\\mu_-|`.
        """
        theta_E = self.einstein_angle()
        u = beta / theta_E
        base = (u**2 + 2.0) / (u * np.sqrt(u**2 + 4.0))
        mu_plus = 0.5 * (base + 1.0)
        mu_minus = 0.5 * (base - 1.0)
        return mu_plus, mu_minus, abs(mu_plus) + abs(mu_minus)


def exact_deflection_angle(bh, impact_parameter, r_far=2.0e5, dtau=None, n_steps=300000):
    """Exact (numerically integrated) Schwarzschild light deflection angle.

    Unlike the weak-field :math:`4M/b` formula used by
    :class:`PointMassLens`, this integrates the actual null geodesic from
    far away, through closest approach, and back out to far away, and
    measures the total bending directly -- valid arbitrarily close to the
    photon sphere (where the deflection formally diverges as
    :math:`b \\to b_c = 3\\sqrt{3}M`).

    Parameters
    ----------
    bh : SchwarzschildBlackHole
        The lensing black hole.
    impact_parameter : float
        Impact parameter :math:`b`; must exceed
        ``bh.critical_impact_parameter`` for the photon to escape.
    r_far : float, default=2e5
        Starting (and target ending) radius, used to approximate "infinity."
    dtau : float, optional
        Affine-parameter step size. Defaults to ``3 * r_far / n_steps``, which
        comfortably covers the round trip to and from ``r_far`` while still
        resolving the near-photon-sphere region for impact parameters not
        too close to critical.
    n_steps : int, default=300000
        Maximum integration steps.

    Returns
    -------
    float
        The deflection angle, in radians.

    Examples
    --------
    >>> from physicskit.relativity.chapters.schwarzschild import SchwarzschildBlackHole
    >>> bh = SchwarzschildBlackHole(M=1.0)
    >>> delta = exact_deflection_angle(bh, impact_parameter=50.0)
    >>> bool(abs(delta - bh.light_deflection_angle(50.0)) < 0.01)
    True
    """
    if dtau is None:
        dtau = 3.0 * r_far / n_steps
    y0 = bh.null_geodesic_initial_state(r0=r_far, impact_parameter=impact_parameter, ingoing=True)
    traj = bh.integrate_geodesic(y0, dtau=dtau, n_steps=n_steps)
    return (traj["phi"][-1] - traj["phi"][0]) - np.pi
