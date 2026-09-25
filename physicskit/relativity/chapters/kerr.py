"""The Kerr black hole: frame dragging, the ergosphere, and the Penrose process.

Roy Kerr's 1963 exact solution describes a rotating black hole, characterized
by mass :math:`M` and spin parameter :math:`a = J/M \\in [0, M)`. Rotation
drags spacetime itself around with it (frame dragging / the Lense-Thirring
effect), creates a region outside the horizon -- the ergosphere -- where no
observer can remain stationary, and permits Penrose's remarkable 1969
proposal: a particle that splits inside the ergosphere can send one fragment
into the black hole with *negative* energy (as measured at infinity), so the
other fragment escapes carrying away more energy than the original particle
had -- extracting rotational energy from the black hole itself.
"""

from __future__ import annotations

import numpy as np

from physicskit.relativity.core.geodesics import integrate_kerr_equatorial_geodesic

__all__ = ["KerrBlackHole"]


class KerrBlackHole:
    """A rotating black hole of mass ``M`` and spin parameter ``a``, in geometrized units.

    Parameters
    ----------
    M : float, default=1.0
        Mass, in geometrized length units.
    a : float, default=0.0
        Spin parameter :math:`a = J/M`, with :math:`0 \\le a < M`.
        ``a=0`` recovers Schwarzschild.
    """

    def __init__(self, M=1.0, a=0.0):
        if not (0.0 <= a < M):
            raise ValueError("spin parameter a must satisfy 0 <= a < M")
        self.M = M
        self.a = a

    @property
    def outer_horizon_radius(self):
        """Event horizon radius, :math:`r_+ = M + \\sqrt{M^2 - a^2}`."""
        return self.M + np.sqrt(self.M**2 - self.a**2)

    @property
    def inner_horizon_radius(self):
        """Inner (Cauchy) horizon radius, :math:`r_- = M - \\sqrt{M^2 - a^2}`."""
        return self.M - np.sqrt(self.M**2 - self.a**2)

    @property
    def dimensionless_spin(self):
        """Dimensionless spin :math:`a_* = a/M \\in [0, 1)`."""
        return self.a / self.M

    def ergosphere_radius(self, theta):
        """Static-limit (ergosphere outer boundary) radius at polar angle ``theta``.

        .. math::

            r_E(\\theta) = M + \\sqrt{M^2 - a^2\\cos^2\\theta}

        The ergosphere is the region :math:`r_+ < r < r_E(\\theta)`, where
        the ``t``-Killing vector becomes spacelike: no observer can remain
        at fixed :math:`(r, \\theta, \\phi)` there, no matter how powerful
        their rocket -- they are inevitably dragged around in :math:`\\phi`
        by the rotating spacetime itself.

        Parameters
        ----------
        theta : float or array_like
            Polar angle (``pi/2`` is the equator, where the ergosphere is
            widest, :math:`r_E = 2M`).

        Returns
        -------
        float or ndarray
        """
        theta = np.asarray(theta, dtype=np.float64)
        return self.M + np.sqrt(self.M**2 - self.a**2 * np.cos(theta) ** 2)

    def is_inside_ergosphere(self, r, theta):
        """Whether a point :math:`(r, \\theta)` lies inside the ergosphere.

        Parameters
        ----------
        r : float
            Radial coordinate.
        theta : float
            Polar angle.

        Returns
        -------
        bool
        """
        return self.outer_horizon_radius < r < self.ergosphere_radius(theta)

    def isco_radius(self, prograde=True):
        """Innermost stable circular equatorial orbit radius (Bardeen-Press-Teukolsky 1972).

        .. math::

            Z_1 = 1 + (1-a_*^2)^{1/3}\\left[(1+a_*)^{1/3} + (1-a_*)^{1/3}\\right], \\quad
            Z_2 = \\sqrt{3a_*^2 + Z_1^2}

        .. math::

            \\frac{r_{\\text{ISCO}}}{M} = 3 + Z_2 \\mp
                \\sqrt{(3-Z_1)(3+Z_1+2Z_2)}

        with the :math:`-` sign for a prograde (co-rotating) orbit and
        :math:`+` for retrograde. Prograde ISCOs shrink toward the horizon
        as spin increases (:math:`r_{\\text{ISCO}} \\to M` as
        :math:`a \\to M`); retrograde ISCOs grow toward :math:`9M`.

        Parameters
        ----------
        prograde : bool, default=True
            Whether the orbit co-rotates with the black hole.

        Returns
        -------
        float
        """
        a_star = self.dimensionless_spin
        Z1 = 1.0 + (1.0 - a_star**2) ** (1.0 / 3.0) * ((1.0 + a_star) ** (1.0 / 3.0) + (1.0 - a_star) ** (1.0 / 3.0))
        Z2 = np.sqrt(3.0 * a_star**2 + Z1**2)
        sign = -1.0 if prograde else 1.0
        return self.M * (3.0 + Z2 + sign * np.sqrt((3.0 - Z1) * (3.0 + Z1 + 2.0 * Z2)))

    def frame_dragging_angular_velocity(self, r):
        """Angular velocity of a zero-angular-momentum observer (ZAMO) in the equatorial plane.

        .. math::

            \\omega(r) = -\\frac{g_{t\\phi}}{g_{\\phi\\phi}}
                       = \\frac{2Ma}{r^3 + a^2 r + 2Ma^2}

        The rate at which locally non-rotating observers are swept around
        in :math:`\\phi` purely by the geometry -- frame dragging. At large
        :math:`r` this falls off as :math:`2Ma/r^3`, the classic
        Lense-Thirring precession rate; it rises sharply approaching the
        ergosphere, where even a photon fired against the rotation is
        dragged forward.

        Parameters
        ----------
        r : float or array_like
            Equatorial radial coordinate, :math:`r > r_+`.

        Returns
        -------
        float or ndarray
        """
        r = np.asarray(r, dtype=np.float64)
        M, a = self.M, self.a
        return 2.0 * M * a / (r**3 + a**2 * r + 2.0 * M * a**2)

    def irreducible_mass(self):
        """Irreducible mass, :math:`M_{\\text{irr}} = \\sqrt{M r_+ / 2}`.

        The irreducible mass can never decrease in any classical process
        (Hawking's area theorem: it is proportional to
        :math:`\\sqrt{\\text{horizon area}}`), which is what caps how much
        rotational energy the Penrose process can extract.

        Returns
        -------
        float
        """
        return np.sqrt(self.M * self.outer_horizon_radius / 2.0)

    def max_penrose_efficiency(self):
        """Maximum fraction of the black hole's mass extractable via the Penrose process.

        .. math::

            \\eta_{\\max} = 1 - \\frac{M_{\\text{irr}}}{M}
                          = 1 - \\sqrt{\\frac{r_+}{2M}}

        Rises from 0 at :math:`a=0` to :math:`1 - 1/\\sqrt{2} \\approx
        29.3\\%` for an extremal (:math:`a \\to M`) black hole.

        Returns
        -------
        float
        """
        return 1.0 - self.irreducible_mass() / self.M

    def penrose_energy_gain(self, initial_energy, fragment_energy_infalling):
        """Energy gained by the escaping fragment in a Penrose-process split.

        A particle of specific energy ``initial_energy`` (measured at
        infinity) splits, inside the ergosphere, into two fragments. If one
        fragment falls into the horizon with negative specific energy
        ``fragment_energy_infalling`` (possible only inside the ergosphere,
        where the ``t``-Killing vector is spacelike), energy conservation at
        the split point requires the escaping fragment to carry away

        .. math::

            E_{\\text{out}} = E_{\\text{in}} - E_{\\text{infalling}} > E_{\\text{in}}

        Parameters
        ----------
        initial_energy : float
            Specific energy of the infalling particle before it splits.
        fragment_energy_infalling : float
            Specific energy of the fragment that falls into the horizon;
            must be negative for energy to be extracted.

        Returns
        -------
        float
            Specific energy of the escaping fragment.

        Examples
        --------
        >>> bh = KerrBlackHole(M=1.0, a=0.9)
        >>> e_out = bh.penrose_energy_gain(initial_energy=1.0, fragment_energy_infalling=-0.05)
        >>> e_out > 1.0
        True
        """
        if fragment_energy_infalling >= 0.0:
            raise ValueError("a Penrose process requires a negative-energy infalling fragment")
        return initial_energy - fragment_energy_infalling

    def circular_orbit_conserved_quantities(self, r0, prograde=True):
        """Conserved specific energy and angular momentum of a circular equatorial orbit.

        Standard Bardeen-Press-Teukolsky (1972) result:

        .. math::

            E = \\frac{r^{3/2} - 2Mr^{1/2} + a M^{1/2}}
                     {r^{3/4}\\sqrt{r^{3/2} - 3Mr^{1/2} + 2aM^{1/2}}}, \\qquad
            L = \\frac{M^{1/2}\\left(r^2 - 2aM^{1/2}r^{1/2} + a^2\\right)}
                     {r^{3/4}\\sqrt{r^{3/2} - 3Mr^{1/2} + 2aM^{1/2}}}

        for a prograde orbit. For a retrograde orbit, substitute
        :math:`a \\to -a` and flip the sign of :math:`L`: it counter-rotates,
        so :math:`L<0` (``d(phi)/d(tau) < 0``) in the hole's frame, which is the
        sign :meth:`integrate_equatorial_geodesic` needs to reproduce the
        circular orbit.

        Parameters
        ----------
        r0 : float
            Orbital radius; should exceed the appropriate
            :meth:`isco_radius` for the orbit to be stable.
        prograde : bool, default=True
            Whether the orbit co-rotates with the black hole.

        Returns
        -------
        E : float
            Specific energy.
        L : float
            Specific angular momentum (negative for a retrograde orbit).
        """
        M = self.M
        a_eff = self.a if prograde else -self.a
        r = r0
        denom = r**0.75 * np.sqrt(r**1.5 - 3.0 * M * r**0.5 + 2.0 * a_eff * M**0.5)
        E = (r**1.5 - 2.0 * M * r**0.5 + a_eff * M**0.5) / denom
        L = np.sqrt(M) * (r**2 - 2.0 * a_eff * np.sqrt(M) * r**0.5 + a_eff**2) / denom
        return E, (L if prograde else -L)

    def integrate_equatorial_geodesic(self, r0, E, L, mu2, dtau, n_steps):
        """Integrate an equatorial Kerr geodesic of conserved energy ``E`` and momentum ``L``.

        Parameters
        ----------
        r0 : float
            Initial radius.
        E : float
            Conserved specific energy.
        L : float
            Conserved specific angular momentum.
        mu2 : float
            ``1.0`` for a timelike geodesic, ``0.0`` for a null (photon) geodesic.
        dtau : float
            Affine-parameter (or proper-time) step size.
        n_steps : int
            Maximum number of steps.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"t"``, ``"r"``, ``"phi"``, ``"ur"``, plus ``"norm"``: the
            four-velocity normalization :math:`g_{\\mu\\nu}u^\\mu u^\\nu`
            at every step (should stay close to :math:`-\\mu^2`; its spread
            measures integration error).
        """
        M, a = self.M, self.a
        Delta0 = r0**2 - 2.0 * M * r0 + a**2
        A0 = E * (r0**2 + a**2) - L * a
        ur0 = np.sqrt(max((A0**2 - Delta0 * (mu2 * r0**2 + (L - a * E) ** 2)) / r0**4, 0.0))
        y0 = np.array([0.0, r0, 0.0, -ur0])

        trajectory, n_valid = integrate_kerr_equatorial_geodesic(y0, E, L, a, M, mu2, dtau, n_steps)
        trajectory = trajectory[: n_valid + 1]
        t, r, phi, ur = trajectory.T

        Delta = r**2 - 2.0 * M * r + a**2
        A = E * (r**2 + a**2) - L * a
        ut = (-a * (a * E - L) + (r**2 + a**2) * A / Delta) / r**2
        uphi = (-(a * E - L) + a * A / Delta) / r**2

        g_tt = -(1.0 - 2.0 * M / r)
        g_tphi = -2.0 * M * a / r
        g_rr = r**2 / Delta
        g_phiphi = r**2 + a**2 + 2.0 * M * a**2 / r
        norm = g_tt * ut**2 + 2.0 * g_tphi * ut * uphi + g_rr * ur**2 + g_phiphi * uphi**2

        return {"t": t, "r": r, "phi": phi, "ur": ur, "ut": ut, "uphi": uphi, "norm": norm}

    def keplerian_angular_velocity(self, r, prograde=True):
        """Coordinate angular velocity of a circular equatorial geodesic orbit.

        .. math::

            \\Omega(r) = \\pm\\frac{\\sqrt{M}}{r^{3/2} \\pm a\\sqrt{M}}

        with the upper sign for a prograde (co-rotating) orbit. Reduces to
        the Keplerian :math:`\\sqrt{M/r^3}` at :math:`a=0`.

        Parameters
        ----------
        r : float or array_like
            Orbital radius.
        prograde : bool, default=True
            Whether the orbit co-rotates with the black hole.

        Returns
        -------
        float or ndarray
        """
        r = np.asarray(r, dtype=np.float64)
        sign = 1.0 if prograde else -1.0
        return sign * np.sqrt(self.M) / (r**1.5 + sign * self.a * np.sqrt(self.M))

    def disk_redshift_factor(self, r, impact_parameter, prograde=True):
        """Combined gravitational + Doppler redshift factor for a circularly orbiting disk emitter.

        Same construction as
        :meth:`physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole.disk_redshift_factor`,
        :math:`g = 1/[u^t(1 - b\\Omega)]`, using the Kerr equatorial metric
        and :meth:`keplerian_angular_velocity` in place of their
        Schwarzschild counterparts.

        Parameters
        ----------
        r : float or array_like
            Disk radius at emission.
        impact_parameter : float or array_like
            The received photon's impact parameter :math:`b = L_{\\text{ph}}/E_{\\text{ph}}`.
        prograde : bool, default=True
            Whether the disk co-rotates with the black hole.

        Returns
        -------
        float or ndarray
            The redshift factor :math:`g`.
        """
        r = np.asarray(r, dtype=np.float64)
        b = np.asarray(impact_parameter, dtype=np.float64)
        M, a = self.M, self.a
        Omega = self.keplerian_angular_velocity(r, prograde=prograde)

        g_tt = -(1.0 - 2.0 * M / r)
        g_tphi = -2.0 * M * a / r
        g_phiphi = r**2 + a**2 + 2.0 * M * a**2 / r
        u_t = 1.0 / np.sqrt(-(g_tt + 2.0 * g_tphi * Omega + g_phiphi * Omega**2))
        return 1.0 / (u_t * (1.0 - b * Omega))
