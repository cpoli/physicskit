"""Binary black hole mergers: the inspiral chirp, plunge, and ringdown.

Two orbiting masses radiate gravitational waves, losing orbital energy and
spiraling closer together, which raises the orbital frequency, which
radiates more strongly still -- a runaway "chirp" that sweeps up in
frequency and amplitude right up until merger. This module implements the
leading-order (Newtonian quadrupole / restricted post-Newtonian) inspiral
waveform -- exactly what let LIGO recognize GW150914 in 2015, the first
direct detection of gravitational waves -- followed by a damped-sinusoid
ringdown as the merged remnant settles down by radiating away its
distortion in a superposition of quasinormal modes.
"""

from __future__ import annotations

import numpy as np

__all__ = ["BinaryMerger"]


class BinaryMerger:
    """A compact binary and its leading-order inspiral-merger-ringdown (IMR) waveform.

    Parameters
    ----------
    m1, m2 : float
        Component masses, in geometrized length units. Use
        :func:`physicskit.relativity.utils.constants.solar_masses_to_geometrized`
        to convert from solar masses. Unlike :class:`~physicskit.relativity.chapters.schwarzschild.SchwarzschildBlackHole`
        or :class:`~physicskit.relativity.chapters.kerr.KerrBlackHole`, there is
        no normalized ``M=1`` convention available here: the radiated
        strain genuinely depends on the absolute physical scale, so an
        unconverted mass doesn't just rescale the answer -- it silently
        produces a wrong waveform. This constructor does *not* itself warn
        on an implausible value (many legitimate uses -- unit tests of a
        formula's mathematical properties, mainly -- pass small
        "toy" numbers on purpose, with no claim of physical realism); use
        :func:`~physicskit.relativity.utils.constants.check_geometrized_mass`
        explicitly wherever real-world realism is actually the point (as
        the visualizers in :mod:`physicskit.relativity.visualizers.wave_plots` do).
    distance : float
        Distance to the source, in geometrized length units. Same caveat
        as ``m1``/``m2``.
    inclination : float, default=0.0
        Inclination angle between the orbital angular momentum and the
        line of sight, in radians (``0`` = face-on, ``pi/2`` = edge-on).
    """

    def __init__(self, m1, m2, distance, inclination=0.0):
        self.m1 = m1
        self.m2 = m2
        self.distance = distance
        self.inclination = inclination

    @property
    def total_mass(self):
        """Total mass, :math:`M = m_1 + m_2`."""
        return self.m1 + self.m2

    @property
    def symmetric_mass_ratio(self):
        """Symmetric mass ratio, :math:`\\eta = m_1 m_2 / M^2`."""
        return self.m1 * self.m2 / self.total_mass**2

    @property
    def chirp_mass(self):
        """Chirp mass, :math:`\\mathcal{M} = (m_1 m_2)^{3/5} / (m_1+m_2)^{1/5}`."""
        return (self.m1 * self.m2) ** 0.6 / self.total_mass**0.2

    def inspiral_frequency(self, t, t_merger):
        """Gravitational-wave frequency during the inspiral, at leading (Newtonian quadrupole) order.

        .. math::

            f(t) = \\frac{1}{\\pi}\\left[\\frac{5}{256(t_{\\text{merger}} - t)}\\right]^{3/8}
                   \\mathcal{M}^{-5/8}

        Parameters
        ----------
        t : float or array_like
            Time(s), with ``t < t_merger``.
        t_merger : float
            Coalescence time.

        Returns
        -------
        float or ndarray
            GW frequency (twice the orbital frequency).
        """
        tau = t_merger - np.asarray(t, dtype=np.float64)
        return (1.0 / np.pi) * (5.0 / (256.0 * tau)) ** 0.375 * self.chirp_mass ** (-5.0 / 8.0)

    def inspiral_phase(self, t, t_merger, phase_ref=0.0):
        """Gravitational-wave phase during the inspiral.

        .. math::

            \\Phi(t) = \\Phi_{\\text{ref}} - 2\\left[\\frac{t_{\\text{merger}} - t}
                       {5\\mathcal{M}}\\right]^{5/8}

        The antiderivative of :math:`d\\Phi/dt = 2\\pi f(t)` with
        :meth:`inspiral_frequency`.

        Parameters
        ----------
        t : float or array_like
            Time(s), with ``t < t_merger``.
        t_merger : float
            Coalescence time.
        phase_ref : float, default=0.0
            Reference phase (phase at merger).

        Returns
        -------
        float or ndarray
        """
        tau = t_merger - np.asarray(t, dtype=np.float64)
        return phase_ref - 2.0 * (tau / (5.0 * self.chirp_mass)) ** (5.0 / 8.0)

    def inspiral_strain(self, t, t_merger, phase_ref=0.0):
        """Plus and cross gravitational-wave strain during the inspiral.

        .. math::

            h_+(t) = \\frac{4}{R} \\mathcal{M}^{5/3} (\\pi f(t))^{2/3}
                     \\frac{1 + \\cos^2\\iota}{2} \\cos\\Phi(t), \\qquad
            h_\\times(t) = \\frac{4}{R} \\mathcal{M}^{5/3} (\\pi f(t))^{2/3}
                     \\cos\\iota \\, \\sin\\Phi(t)

        Parameters
        ----------
        t : array_like
            Times, with ``t < t_merger``.
        t_merger : float
            Coalescence time.
        phase_ref : float, default=0.0
            Reference phase at merger, forwarded to :meth:`inspiral_phase`.

        Returns
        -------
        h_plus : ndarray
        h_cross : ndarray

        Examples
        --------
        >>> merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
        >>> t = np.linspace(-2.0, -0.1, 100)
        >>> hp, hc = merger.inspiral_strain(t, t_merger=0.0)
        >>> hp.shape
        (100,)
        """
        t = np.asarray(t, dtype=np.float64)
        f = self.inspiral_frequency(t, t_merger)
        phase = self.inspiral_phase(t, t_merger, phase_ref=phase_ref)
        amplitude = (4.0 / self.distance) * self.chirp_mass ** (5.0 / 3.0) * (np.pi * f) ** (2.0 / 3.0)
        h_plus = amplitude * 0.5 * (1.0 + np.cos(self.inclination) ** 2) * np.cos(phase)
        h_cross = amplitude * np.cos(self.inclination) * np.sin(phase)
        return h_plus, h_cross

    def remnant_estimate(self):
        """A simple, illustrative estimate of the merger remnant's mass and spin.

        Not a precision numerical-relativity surrogate: it linearly
        interpolates between the extreme-mass-ratio limit (negligible
        radiated energy and spin-up, :math:`\\eta \\to 0`) and the
        well-measured equal-mass, non-spinning result
        (:math:`M_f \\approx 0.95 M`, :math:`a_f/M_f \\approx 0.69` at
        :math:`\\eta = 1/4`), which is sufficient to demonstrate the
        qualitative ringdown physics.

        Returns
        -------
        M_f : float
            Estimated remnant mass.
        a_f : float
            Estimated remnant spin parameter.
        """
        eta = self.symmetric_mass_ratio
        M_f = self.total_mass * (1.0 - 0.05 * (4.0 * eta))
        a_f = 0.69 * (4.0 * eta) * M_f
        return M_f, a_f

    def qnm_frequency_damping(self, M_f=None, a_f=None):
        """Dominant (l=2, m=2, n=0) quasinormal mode frequency and damping time.

        Fitting formulas of Berti, Cardoso & Will (2006):

        .. math::

            f_{\\text{QNM}} = \\frac{1}{2\\pi M_f}\\left[1.5251 - 1.1568(1-a_*)^{0.1292}\\right]

        .. math::

            Q = 0.7000 + 1.4187(1-a_*)^{-0.4990}, \\qquad
            \\tau_{\\text{damp}} = \\frac{Q}{\\pi f_{\\text{QNM}}}

        where :math:`a_* = a_f/M_f`.

        Parameters
        ----------
        M_f : float, optional
            Remnant mass. Defaults to :meth:`remnant_estimate`.
        a_f : float, optional
            Remnant spin parameter. Defaults to :meth:`remnant_estimate`.

        Returns
        -------
        f_qnm : float
            Ringdown frequency.
        tau_damp : float
            Damping time.
        """
        if M_f is None or a_f is None:
            M_f, a_f = self.remnant_estimate()
        a_star = a_f / M_f
        f_qnm = (1.0 / (2.0 * np.pi * M_f)) * (1.5251 - 1.1568 * (1.0 - a_star) ** 0.1292)
        Q = 0.7000 + 1.4187 * (1.0 - a_star) ** (-0.4990)
        tau_damp = Q / (np.pi * f_qnm)
        return f_qnm, tau_damp

    def ringdown_strain(self, t, t_merger, amplitude, M_f=None, a_f=None, phase_ref=0.0):
        """Damped-sinusoid ringdown strain for :math:`t \\ge t_{\\text{merger}}`.

        .. math::

            h_+(t) = A \\, e^{-(t-t_{\\text{merger}})/\\tau_{\\text{damp}}}
                       \\cos\\left(2\\pi f_{\\text{QNM}} (t - t_{\\text{merger}}) + \\Phi_{\\text{ref}}\\right)

        with :math:`h_\\times` the corresponding sine (quarter-cycle
        phase-shifted) quadrature, modulated by the same inclination
        dependence as :meth:`inspiral_strain`.

        Parameters
        ----------
        t : array_like
            Times, with ``t >= t_merger``.
        t_merger : float
            Merger (ringdown onset) time.
        amplitude : float
            Ringdown strain amplitude at :math:`t = t_{\\text{merger}}`,
            typically chosen to match the inspiral amplitude just before
            merger for a continuous (if not perfectly smooth) toy waveform.
        M_f, a_f : float, optional
            Remnant mass and spin, forwarded to :meth:`qnm_frequency_damping`.
        phase_ref : float, default=0.0
            Reference phase at :math:`t = t_{\\text{merger}}`.

        Returns
        -------
        h_plus : ndarray
        h_cross : ndarray
        """
        t = np.asarray(t, dtype=np.float64)
        f_qnm, tau_damp = self.qnm_frequency_damping(M_f=M_f, a_f=a_f)
        dt = t - t_merger
        envelope = amplitude * np.exp(-dt / tau_damp)
        phase = 2.0 * np.pi * f_qnm * dt + phase_ref
        h_plus = envelope * 0.5 * (1.0 + np.cos(self.inclination) ** 2) * np.cos(phase)
        h_cross = envelope * np.cos(self.inclination) * np.sin(phase)
        return h_plus, h_cross

    def full_waveform(self, t, t_merger):
        """A continuous inspiral-merger-ringdown toy waveform over an arbitrary time array.

        Uses :meth:`inspiral_strain` for ``t < t_merger`` and
        :meth:`ringdown_strain` for ``t >= t_merger``, with the ringdown
        amplitude anchored to the inspiral amplitude evaluated just before
        merger for continuity.

        Parameters
        ----------
        t : array_like
            Time array, spanning both before and after ``t_merger``.
        t_merger : float
            Coalescence time; should be strictly greater than every
            pre-merger sample (the inspiral formula diverges at
            ``t = t_merger``).

        Returns
        -------
        h_plus : ndarray
        h_cross : ndarray

        Examples
        --------
        >>> merger = BinaryMerger(m1=30.0, m2=25.0, distance=1000.0)
        >>> t = np.linspace(-1.0, 0.02, 500)
        >>> hp, hc = merger.full_waveform(t, t_merger=0.0)
        >>> hp.shape
        (500,)
        """
        t = np.asarray(t, dtype=np.float64)
        h_plus = np.zeros_like(t)
        h_cross = np.zeros_like(t)

        inspiral_mask = t < t_merger
        ringdown_mask = ~inspiral_mask

        if np.any(inspiral_mask):
            hp, hc = self.inspiral_strain(t[inspiral_mask], t_merger)
            h_plus[inspiral_mask] = hp
            h_cross[inspiral_mask] = hc

        t_match = t_merger - 1.0e-3 * abs(t_merger if t_merger != 0 else 1.0)
        hp_match, hc_match = self.inspiral_strain(np.array([t_match]), t_merger)
        match_amplitude = float(np.hypot(hp_match[0], hc_match[0]))

        if np.any(ringdown_mask):
            hp, hc = self.ringdown_strain(t[ringdown_mask], t_merger, amplitude=match_amplitude)
            h_plus[ringdown_mask] = hp
            h_cross[ringdown_mask] = hc

        return h_plus, h_cross

    def orbital_period(self, semi_major_axis):
        """Newtonian (Keplerian) orbital period at a given semi-major axis.

        .. math::

            P = 2\\pi \\sqrt{\\frac{a^3}{M}}

        Valid for the wide, slowly-decaying orbits this method's
        counterparts describe (unlike :meth:`inspiral_frequency`, which
        covers the final relativistic plunge).

        Parameters
        ----------
        semi_major_axis : float
            Orbital semi-major axis :math:`a`.

        Returns
        -------
        float
        """
        return 2.0 * np.pi * np.sqrt(semi_major_axis**3 / self.total_mass)

    def semi_major_axis_decay_rate(self, semi_major_axis, eccentricity):
        """Rate of orbital shrinkage from gravitational-wave emission (Peters 1964).

        .. math::

            \\frac{da}{dt} = -\\frac{64}{5} \\frac{m_1 m_2 (m_1+m_2)}{a^3}
                \\frac{1 + \\frac{73}{24}e^2 + \\frac{37}{96}e^4}{(1-e^2)^{7/2}}

        This is the formula behind the first (indirect) detection of
        gravitational waves: Hulse and Taylor's 1974 discovery of the
        binary pulsar PSR B1913+16, whose orbital period was found to
        decay at precisely the rate this equation predicts (1993 Nobel
        Prize in Physics).

        Parameters
        ----------
        semi_major_axis : float
            Orbital semi-major axis :math:`a`.
        eccentricity : float
            Orbital eccentricity :math:`e \\in [0, 1)`.

        Returns
        -------
        float
            :math:`da/dt` (negative: the orbit shrinks).
        """
        e = eccentricity
        enhancement = (1.0 + 73.0 / 24.0 * e**2 + 37.0 / 96.0 * e**4) / (1.0 - e**2) ** 3.5
        return -64.0 / 5.0 * self.m1 * self.m2 * self.total_mass / semi_major_axis**3 * enhancement

    def period_decay_rate(self, semi_major_axis, eccentricity):
        """Rate of orbital period decay, :math:`dP/dt`, from gravitational-wave emission.

        Obtained from :meth:`semi_major_axis_decay_rate` via the chain rule
        through Kepler's third law, :math:`dP/dt = (dP/da)(da/dt)` with
        :math:`dP/da = 3\\pi\\sqrt{a/M}`.

        Parameters
        ----------
        semi_major_axis : float
            Orbital semi-major axis :math:`a`.
        eccentricity : float
            Orbital eccentricity :math:`e \\in [0, 1)`.

        Returns
        -------
        float
            :math:`dP/dt` (dimensionless: a rate of change of time per unit
            time; negative, since the period shrinks).

        Examples
        --------
        >>> import physicskit.relativity.utils.constants as const
        >>> m1 = const.solar_masses_to_geometrized(1.4398)
        >>> m2 = const.solar_masses_to_geometrized(1.3886)
        >>> psr_b1913 = BinaryMerger(m1=m1, m2=m2, distance=1.0)
        >>> a = (psr_b1913.total_mass * (27906.98 * const.C_SI) ** 2 / (4 * np.pi**2)) ** (1 / 3)
        >>> dPdt = psr_b1913.period_decay_rate(a, eccentricity=0.6171338)
        >>> bool(-3e-12 < dPdt < -2e-12)  # observed: -2.4e-12 (dimensionless)
        True
        """
        dP_da = 3.0 * np.pi * np.sqrt(semi_major_axis / self.total_mass)
        return dP_da * self.semi_major_axis_decay_rate(semi_major_axis, eccentricity)
