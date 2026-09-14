"""Neutron star structure: the Tolman-Oppenheimer-Volkoff equations.

A black hole is not the only fate of a massive star's core: below about
:math:`3 M_\\odot`, degenerate neutron pressure can halt gravitational
collapse entirely, leaving behind a neutron star -- matter compressed to
nuclear density and held up against its own gravity by quantum degeneracy
pressure. Oppenheimer and Volkoff (1939), building on Tolman's earlier work,
derived the relativistic generalization of the Newtonian hydrostatic
equilibrium equation for a self-gravitating fluid sphere. Unlike the
Newtonian case, the TOV equation predicts a maximum possible neutron star
mass: add too much matter and pressure alone can no longer resist collapse,
regardless of the equation of state -- the star must become a black hole.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

__all__ = ["NeutronStar"]


class NeutronStar:
    """A neutron star modeled with a polytropic equation of state, solved via the TOV equations.

    The equation of state is the polytrope :math:`P = K \\rho_0^\\Gamma`
    relating pressure to rest-mass density; the total energy density
    (rest mass plus internal energy) used as the gravitational source is
    :math:`\\varepsilon = \\rho_0 + P/(\\Gamma - 1)`. All quantities are in
    geometrized units; ``K`` and ``Gamma`` are illustrative "toy" parameters
    unless calibrated to a specific realistic equation of state.

    Parameters
    ----------
    K : float, default=100.0
        Polytropic constant.
    Gamma : float, default=2.0
        Polytropic index (:math:`\\Gamma = 2` is a common toy choice
        loosely representative of nuclear matter).
    """

    def __init__(self, K=100.0, Gamma=2.0):
        self.K = K
        self.Gamma = Gamma

    def energy_density(self, pressure):
        """Total energy density :math:`\\varepsilon(P) = (P/K)^{1/\\Gamma} + P/(\\Gamma-1)`.

        Parameters
        ----------
        pressure : float or array_like
            Pressure.

        Returns
        -------
        float or ndarray
        """
        pressure = np.asarray(pressure, dtype=np.float64)
        rest_mass_density = (pressure / self.K) ** (1.0 / self.Gamma)
        return rest_mass_density + pressure / (self.Gamma - 1.0)

    def _tov_rhs(self, r, y):
        m, P = y
        if P <= 0.0:
            return [0.0, 0.0]
        eps = self.energy_density(P)
        dm_dr = 4.0 * np.pi * r**2 * eps
        dP_dr = -(eps + P) * (m + 4.0 * np.pi * r**3 * P) / (r * (r - 2.0 * m))
        return [dm_dr, dP_dr]

    def solve(self, central_density, r_max=50.0, r_start=1.0e-6):
        """Integrate the TOV equations outward from the center to the stellar surface.

        Parameters
        ----------
        central_density : float
            Central rest-mass density :math:`\\rho_0(0)`.
        r_max : float, default=50.0
            Outer integration bound (should exceed any physically expected
            stellar radius for the given equation of state).
        r_start : float, default=1e-6
            Small starting radius (avoids the coordinate singularity at
            exactly :math:`r=0`).

        Returns
        -------
        mass : float
            Total gravitational mass :math:`M`, read off where the pressure
            drops to (numerically) zero -- the stellar surface.
        radius : float
            Stellar radius :math:`R`.
        solution : scipy.integrate.OdeSolution
            The full ``solve_ivp`` result, for inspecting the interior
            profile.

        Examples
        --------
        >>> star = NeutronStar(K=100.0, Gamma=2.0)
        >>> M, R, sol = star.solve(central_density=1.28e-3)
        >>> bool(M > 0.0 and R > 0.0)
        True
        """
        pressure_central = self.K * central_density**self.Gamma

        def surface_event(r, y):
            return y[1] - 1.0e-10 * pressure_central

        surface_event.terminal = True
        surface_event.direction = -1

        solution = solve_ivp(
            self._tov_rhs,
            [r_start, r_max],
            [0.0, pressure_central],
            events=surface_event,
            dense_output=True,
            max_step=0.01,
            rtol=1.0e-8,
            atol=1.0e-14,
        )
        radius = solution.t[-1]
        mass = solution.y[0, -1]
        return mass, radius, solution

    def mass_radius_curve(self, central_densities):
        """Trace the mass-radius relation over a range of central densities.

        The existence of a maximum mass (a turning point in :math:`M` as a
        function of central density, beyond which increasing the central
        density *decreases* the stable mass) is the TOV equation's central
        physical prediction.

        Parameters
        ----------
        central_densities : array_like
            Central densities to sample.

        Returns
        -------
        masses : ndarray
        radii : ndarray
        """
        masses = np.empty(len(central_densities))
        radii = np.empty(len(central_densities))
        for i, rho_c in enumerate(central_densities):
            M, R, _ = self.solve(rho_c)
            masses[i] = M
            radii[i] = R
        return masses, radii
