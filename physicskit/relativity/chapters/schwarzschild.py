"""The Schwarzschild black hole: orbits, precession, light bending, and the photon sphere.

The Schwarzschild solution -- the first exact solution to Einstein's field
equations, found by Karl Schwarzschild in 1916 while serving on the Russian
front, just weeks after Einstein published General Relativity -- describes
the spacetime around any non-rotating, uncharged spherical mass. It predicts
three of GR's classic observational triumphs: the anomalous perihelion
precession of Mercury, the bending of starlight grazing the Sun (confirmed
by Eddington's 1919 eclipse expedition), and the Shapiro time delay of radar
signals passing near the Sun.
"""

from __future__ import annotations

import numpy as np

from physicskit.relativity.core.geodesics import integrate_schwarzschild_geodesic

__all__ = ["SchwarzschildBlackHole"]


class SchwarzschildBlackHole:
    """A non-rotating black hole (or any spherical mass) of mass ``M``, in geometrized units.

    Parameters
    ----------
    M : float, default=1.0
        Mass, in geometrized length units (:math:`GM/c^2`). Use
        :func:`physicskit.relativity.utils.constants.solar_masses_to_geometrized`
        to convert from a physical mass in solar masses.
    """

    def __init__(self, M=1.0):
        self.M = M

    @property
    def horizon_radius(self):
        """Event horizon radius, :math:`r_s = 2M`."""
        return 2.0 * self.M

    @property
    def photon_sphere_radius(self):
        """Photon sphere radius, :math:`r_{ph} = 3M`."""
        return 3.0 * self.M

    @property
    def isco_radius(self):
        """Innermost stable circular orbit radius, :math:`r_{\\text{ISCO}} = 6M`."""
        return 6.0 * self.M

    @property
    def critical_impact_parameter(self):
        """Critical photon impact parameter, :math:`b_c = 3\\sqrt{3}\\,M`."""
        return 3.0 * np.sqrt(3.0) * self.M

    def circular_orbit_angular_velocity(self, r):
        """Coordinate angular velocity :math:`\\Omega = d\\phi/dt` of a circular timelike orbit.

        Exactly Keplerian in Schwarzschild coordinates:
        :math:`\\Omega = \\sqrt{M/r^3}`.

        Parameters
        ----------
        r : float
            Orbital radius.

        Returns
        -------
        float
        """
        return np.sqrt(self.M / r**3)

    def circular_orbit_initial_state(self, r0):
        """Initial 8-component state for a circular, equatorial, timelike orbit at radius ``r0``.

        Parameters
        ----------
        r0 : float
            Orbital radius. Should exceed :attr:`isco_radius` for the orbit
            to be stable.

        Returns
        -------
        ndarray of shape (8,)
            :math:`(t, r, \\theta, \\phi, u^t, u^r, u^\\theta, u^\\phi)` at
            :math:`\\tau = 0`.
        """
        M = self.M
        f = 1.0 - 2.0 * M / r0
        Omega = self.circular_orbit_angular_velocity(r0)
        ut = 1.0 / np.sqrt(f - r0**2 * Omega**2)
        uphi = Omega * ut
        return np.array([0.0, r0, np.pi / 2.0, 0.0, ut, 0.0, 0.0, uphi])

    def eccentric_orbit_initial_state(self, r0, eccentricity_boost=0.0):
        """Initial state for a bound, generally eccentric equatorial timelike orbit.

        Starts at radius ``r0`` (taken as the initial apoapsis) with purely
        tangential velocity reduced below the circular value by
        ``eccentricity_boost``, producing a radially oscillating (rosette,
        precessing) orbit for ``eccentricity_boost > 0``.

        Parameters
        ----------
        r0 : float
            Starting radius (apoapsis).
        eccentricity_boost : float, default=0.0
            Fractional reduction of the tangential velocity relative to the
            circular orbit at ``r0``; ``0`` gives a circular orbit, larger
            values give more eccentric orbits.

        Returns
        -------
        ndarray of shape (8,)
            Initial state, as in :meth:`circular_orbit_initial_state`.
        """
        M = self.M
        f = 1.0 - 2.0 * M / r0
        Omega = self.circular_orbit_angular_velocity(r0) * (1.0 - eccentricity_boost)
        ut = 1.0 / np.sqrt(f - r0**2 * Omega**2)
        uphi = Omega * ut
        return np.array([0.0, r0, np.pi / 2.0, 0.0, ut, 0.0, 0.0, uphi])

    def null_geodesic_initial_state(self, r0, impact_parameter, ingoing=True):
        """Initial state for an equatorial null (photon) geodesic with a given impact parameter.

        Parameters
        ----------
        r0 : float
            Starting radius.
        impact_parameter : float
            Impact parameter :math:`b = L/E`.
        ingoing : bool, default=True
            Whether the photon starts moving toward decreasing :math:`r`.

        Returns
        -------
        ndarray of shape (8,)
            Initial state with :math:`E = 1` (an affine-parameter
            normalization choice; only the ratio :math:`b` is physical).
        """
        M = self.M
        f = 1.0 - 2.0 * M / r0
        E = 1.0
        L = impact_parameter * E
        ut = E / f
        uphi = L / r0**2
        radicand = E**2 - f * L**2 / r0**2
        ur = -np.sqrt(max(radicand, 0.0)) if ingoing else np.sqrt(max(radicand, 0.0))
        return np.array([0.0, r0, np.pi / 2.0, 0.0, ut, ur, 0.0, uphi])

    def integrate_geodesic(self, y0, dtau, n_steps):
        """Integrate a geodesic from initial state ``y0`` via fixed-step RK4.

        Parameters
        ----------
        y0 : ndarray of shape (8,)
            Initial state, e.g. from :meth:`circular_orbit_initial_state`,
            :meth:`eccentric_orbit_initial_state`, or
            :meth:`null_geodesic_initial_state`.
        dtau : float
            Affine-parameter (or proper-time) step size.
        n_steps : int
            Maximum number of steps.

        Returns
        -------
        dict of str -> ndarray
            Keys ``"t"``, ``"r"``, ``"theta"``, ``"phi"``, ``"ut"``,
            ``"ur"``, ``"utheta"``, ``"uphi"``, each a 1D array over the
            (trimmed) trajectory, plus ``"energy"`` and
            ``"angular_momentum"``: the conserved
            :math:`E = -u_t = (1-2M/r) u^t` and
            :math:`L = u_\\phi = r^2 \\sin^2\\theta \\, u^\\phi` evaluated
            at every step (constant along an exact geodesic; their spread
            measures integration error).

        Examples
        --------
        >>> bh = SchwarzschildBlackHole(M=1.0)
        >>> y0 = bh.circular_orbit_initial_state(r0=10.0)
        >>> traj = bh.integrate_geodesic(y0, dtau=0.01, n_steps=500)
        >>> float(np.std(traj["energy"])) < 1e-6
        True
        """
        trajectory, n_valid = integrate_schwarzschild_geodesic(y0, self.M, dtau, n_steps)
        trajectory = trajectory[: n_valid + 1]
        t, r, theta, phi, ut, ur, uth, uphi = trajectory.T
        f = 1.0 - 2.0 * self.M / r
        energy = f * ut
        ang_mom = r**2 * np.sin(theta) ** 2 * uphi
        return {
            "t": t,
            "r": r,
            "theta": theta,
            "phi": phi,
            "ut": ut,
            "ur": ur,
            "utheta": uth,
            "uphi": uphi,
            "energy": energy,
            "angular_momentum": ang_mom,
        }

    def perihelion_precession(self, trajectory):
        """Measure the apsidal (perihelion) precession per orbit from an integrated trajectory.

        Locates successive periapsis passages (local minima of :math:`r(\\lambda)`)
        and returns the excess azimuthal angle swept between them beyond
        :math:`2\\pi`.

        Parameters
        ----------
        trajectory : dict of str -> ndarray
            Output of :meth:`integrate_geodesic` for an eccentric orbit.

        Returns
        -------
        ndarray
            Precession (in radians) measured between each pair of
            consecutive periapsis passages found in the trajectory.
        """
        r = trajectory["r"]
        phi = trajectory["phi"]
        is_min = (r[1:-1] < r[:-2]) & (r[1:-1] < r[2:])
        idx = np.nonzero(is_min)[0] + 1
        if len(idx) < 2:
            return np.array([])
        phi_at_peri = phi[idx]
        delta_phi = np.diff(phi_at_peri)
        return delta_phi - 2.0 * np.pi

    def weak_field_precession_per_orbit(self, semi_major_axis, eccentricity):
        """Leading-order post-Newtonian perihelion precession per orbit.

        .. math::

            \\Delta\\phi = \\frac{6\\pi M}{a (1 - e^2)}

        Parameters
        ----------
        semi_major_axis : float
            Orbital semi-major axis :math:`a`.
        eccentricity : float
            Orbital eccentricity :math:`e \\in [0, 1)`.

        Returns
        -------
        float
            Precession per orbit, in radians.
        """
        return 6.0 * np.pi * self.M / (semi_major_axis * (1.0 - eccentricity**2))

    def light_deflection_angle(self, impact_parameter):
        """Leading-order light deflection angle for a photon passing at a given impact parameter.

        .. math::

            \\delta\\phi \\approx \\frac{4M}{b}

        The weak-field (:math:`b \\gg M`) limit of gravitational light
        bending; famously confirmed for starlight grazing the Sun during
        the 1919 solar eclipse.

        Parameters
        ----------
        impact_parameter : float
            Impact parameter :math:`b`.

        Returns
        -------
        float
            Deflection angle, in radians.
        """
        return 4.0 * self.M / impact_parameter

    def shapiro_delay(self, r1, r2, impact_parameter):
        """Shapiro (gravitational) time delay for light passing between two radii.

        .. math::

            \\Delta t = 2M \\ln\\left[
                \\frac{\\left(r_1 + \\sqrt{r_1^2 - b^2}\\right)
                       \\left(r_2 + \\sqrt{r_2^2 - b^2}\\right)}{b^2}
            \\right]

        The excess light-travel time (beyond flat-spacetime expectations)
        caused by spacetime curvature, first measured via radar ranging to
        Venus and the Viking Mars landers.

        Parameters
        ----------
        r1, r2 : float
            Radial coordinates of the two endpoints (e.g. Earth and a
            reflecting planet or spacecraft).
        impact_parameter : float
            Closest approach distance of the light path to the mass.

        Returns
        -------
        float
            Time delay, in geometrized time units (length; divide by
            :data:`physicskit.relativity.utils.constants.C_SI` for seconds).
        """
        b = impact_parameter
        term1 = r1 + np.sqrt(r1**2 - b**2)
        term2 = r2 + np.sqrt(r2**2 - b**2)
        return 2.0 * self.M * np.log(term1 * term2 / b**2)

    def effective_potential(self, r, L):
        """Effective potential for equatorial timelike orbits, :math:`V_{\\text{eff}}(r) = (1-2M/r)(1+L^2/r^2)`.

        Radial motion obeys :math:`(dr/d\\tau)^2 = E^2 - V_{\\text{eff}}(r)`.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate(s).
        L : float
            Specific angular momentum.

        Returns
        -------
        float or ndarray
        """
        r = np.asarray(r, dtype=np.float64)
        return (1.0 - 2.0 * self.M / r) * (1.0 + L**2 / r**2)

    def disk_redshift_factor(self, r, impact_parameter):
        """Combined gravitational + Doppler redshift factor for a circularly orbiting disk emitter.

        For a photon with impact parameter :math:`b = L_{\\text{ph}}/E_{\\text{ph}}`
        received from gas in a circular Keplerian orbit at radius :math:`r`,
        the ratio of observed to emitted photon energy is

        .. math::

            g = \\frac{E_{\\text{obs}}}{E_{\\text{emit}}}
              = \\frac{1}{u^t(r)\\left[1 - b\\,\\Omega(r)\\right]}

        where :math:`u^t(r) = 1/\\sqrt{f(r) - r^2\\Omega(r)^2}` is the
        emitter's time dilation factor and :math:`\\Omega(r) =
        \\sqrt{M/r^3}` its orbital angular velocity (see
        :meth:`circular_orbit_angular_velocity`). :math:`g < 1` is redshifted
        (gravity, or the far/receding side of the disk); :math:`g > 1` is
        blueshifted (the near/approaching side, when Doppler beaming wins).

        Parameters
        ----------
        r : float or array_like
            Disk radius at emission.
        impact_parameter : float or array_like
            The received photon's impact parameter :math:`b`.

        Returns
        -------
        float or ndarray
            The redshift factor :math:`g`.
        """
        r = np.asarray(r, dtype=np.float64)
        b = np.asarray(impact_parameter, dtype=np.float64)
        f = 1.0 - 2.0 * self.M / r
        Omega = self.circular_orbit_angular_velocity(r)
        u_t = 1.0 / np.sqrt(f - r**2 * Omega**2)
        return 1.0 / (u_t * (1.0 - b * Omega))

    def tidal_acceleration(self, r, proper_separation):
        """Tidal (geodesic deviation) acceleration on a radially infalling object.

        In the local orthonormal frame of a freely-falling observer, the
        Riemann tensor's nonzero components give a stretching acceleration
        along the radial direction and an equal compression along each
        transverse direction:

        .. math::

            a_{\\text{radial}} = \\frac{2M}{r^3}\\,\\ell, \\qquad
            a_{\\text{transverse}} = -\\frac{M}{r^3}\\,\\ell

        for two points separated by a small proper length :math:`\\ell`.
        This is "spaghettification": an object falling feet-first is
        stretched head-to-toe and squeezed side-to-side, with both effects
        growing as :math:`1/r^3` and formally diverging at the singularity.

        Parameters
        ----------
        r : float or array_like
            Radial coordinate.
        proper_separation : float
            Proper length between the two points being torn apart (e.g. the
            height of an infalling astronaut).

        Returns
        -------
        radial : float or ndarray
            Stretching acceleration along the radial direction (positive:
            pulls the two points apart).
        transverse : float or ndarray
            Compressing acceleration along each transverse direction
            (negative: pushes together).

        Examples
        --------
        >>> bh = SchwarzschildBlackHole(M=1.0)
        >>> radial, transverse = bh.tidal_acceleration(r=6.0, proper_separation=1.0)
        >>> bool(radial > 0.0 and transverse < 0.0)
        True
        """
        r = np.asarray(r, dtype=np.float64)
        radial = 2.0 * self.M / r**3 * proper_separation
        transverse = -self.M / r**3 * proper_separation
        return radial, transverse
