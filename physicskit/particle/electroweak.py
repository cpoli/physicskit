r"""Toy models of three electroweak/QED phenomena: spontaneous symmetry
breaking, leading-order QED annihilation, and neutral-meson CP violation.

Uses the same natural-unit convention (:math:`\hbar=c=1`) as
:mod:`physicskit.particle.kinematics`.

- :func:`higgs_potential`, :func:`higgs_vev`, :func:`higgs_field_rollover`
  -- classical spontaneous symmetry breaking of a real scalar field in a
  :math:`\phi^4` double-well potential (the same mechanism underlying
  the Higgs field's nonzero vacuum expectation value), reduced to a
  single spatial point (0+1 dimensional) for simplicity.
- :func:`qed_dsigma_domega_mumu`, :func:`qed_total_cross_section_mumu` --
  the leading-order (tree-level, single-photon-exchange) QED
  differential and total cross sections for :math:`e^+e^-\to\mu^+\mu^-`.
- :func:`meson_decay_rates_cp_eigenstate`, :func:`cp_asymmetry` -- a
  Wigner-Weisskopf two-state-mixing toy model of neutral-meson
  (:math:`K^0`-:math:`\bar K^0`-style) CP violation in the decay-rate
  asymmetry to a common CP eigenstate final state.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import solve_ivp

from physicskit.particle.scattering import ALPHA_FS

__all__ = [
    "higgs_potential",
    "higgs_vev",
    "higgs_field_rollover",
    "qed_dsigma_domega_mumu",
    "qed_total_cross_section_mumu",
    "meson_decay_rates_cp_eigenstate",
    "cp_asymmetry",
]


def higgs_potential(phi, a, b):
    r"""The classical :math:`\phi^4` double-well potential, :math:`V(\phi)=-a\phi^2+b\phi^4`.

    For :math:`a,b>0` this has an unstable extremum at :math:`\phi=0`
    (the symmetric, "false vacuum" point) and two degenerate true minima
    at :math:`\phi=\pm v`, :math:`v=\sqrt{a/(2b)}` -- the field must
    "choose" one of them, spontaneously breaking the :math:`\phi\to-\phi`
    symmetry. This is the same quartic potential that gives the Higgs
    field its nonzero vacuum expectation value.

    Parameters
    ----------
    phi : array_like
        Field value.
    a, b : float
        Potential parameters, :math:`a,b>0`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> float(higgs_potential(0.0, a=1.0, b=1.0))
    0.0
    """
    phi = np.asarray(phi, dtype=float)
    return -a * phi**2 + b * phi**4


def higgs_vev(a, b):
    r"""The true-vacuum field value, :math:`v=\sqrt{a/(2b)}`, minimizing :func:`higgs_potential`.

    Examples
    --------
    >>> round(higgs_vev(a=2.0, b=1.0), 6)
    1.0
    """
    return float(np.sqrt(a / (2.0 * b)))


def higgs_field_rollover(phi0, phidot0, a, b, t_eval, damping=0.0):
    r"""Classically evolve a homogeneous scalar field rolling in :func:`higgs_potential`.

    Integrates the (0+1 dimensional -- no spatial gradient/kink
    structure, a single point standing in for a spatially uniform field
    configuration) classical field equation

    .. math::

        \ddot\phi = -\frac{dV}{d\phi} - \gamma\dot\phi
        = 2a\phi - 4b\phi^3 - \gamma\dot\phi

    Started near the unstable symmetric point :math:`\phi\approx0` with
    a tiny perturbation, the field is repelled from it and rolls down
    into one of the two true vacua :math:`\phi=\pm v`
    (:func:`higgs_vev`) -- whichever side the initial perturbation
    pushes it toward, a toy realization of spontaneous symmetry
    breaking. A small damping :math:`\gamma>0` (representing energy
    radiated into other field modes, not modeled explicitly) lets the
    field settle into the chosen vacuum instead of oscillating in it
    forever.

    Parameters
    ----------
    phi0, phidot0 : float
        Initial field value and velocity.
    a, b : float
        Potential parameters, see :func:`higgs_potential`.
    t_eval : array_like
        Times at which to report the solution.
    damping : float, default=0.0
        Damping coefficient :math:`\gamma\ge0`.

    Returns
    -------
    phi, phidot : ndarray
        Field value and velocity at each time in ``t_eval``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 60, 600)
    >>> phi, phidot = higgs_field_rollover(1e-3, 0.0, a=1.0, b=1.0, t_eval=t, damping=0.08)
    >>> v = higgs_vev(1.0, 1.0)
    >>> bool(abs(abs(phi[-1]) - v) < 0.05)
    True
    """
    t_eval = np.asarray(t_eval, dtype=float)

    def rhs(t, y):
        phi, phidot = y
        return [phidot, 2.0 * a * phi - 4.0 * b * phi**3 - damping * phidot]

    sol = solve_ivp(rhs, (t_eval[0], t_eval[-1]), [phi0, phidot0], t_eval=t_eval, rtol=1e-9, atol=1e-12)
    return sol.y[0], sol.y[1]


def qed_dsigma_domega_mumu(cos_theta, sqrt_s, alpha=ALPHA_FS):
    r"""Leading-order QED differential cross section for :math:`e^+e^-\to\mu^+\mu^-`.

    .. math::

        \frac{d\sigma}{d\Omega} = \frac{\alpha^2}{4s}\left(1+\cos^2\theta\right)

    the standard tree-level, single-photon-exchange, unpolarized result
    in the ultrarelativistic (massless-fermion) limit -- e.g. Peskin &
    Schroeder, *An Introduction to Quantum Field Theory*, eq. (5.15); or
    Halzen & Martin, *Quarks and Leptons*, eq. (6.23). :math:`\theta` is
    the muon's polar angle relative to the beam in the center-of-mass
    frame, and :math:`s=(\sqrt s)^2` is the Mandelstam invariant (see
    :func:`physicskit.particle.scattering.mandelstam_s`).

    **Simplification**: electron and muon masses are neglected (valid
    for :math:`\sqrt s\gg2m_\mu\approx0.2` GeV); higher-order QED
    corrections and any :math:`Z`-boson contribution (relevant near and
    above the :math:`Z` pole, :math:`\sqrt s\sim91` GeV) are not included.

    Parameters
    ----------
    cos_theta : array_like
        Cosine of the muon's center-of-mass polar angle.
    sqrt_s : float
        Center-of-mass energy.
    alpha : float, default=ALPHA_FS
        Fine-structure constant.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(qed_dsigma_domega_mumu(0.0, sqrt_s=10.0) / qed_dsigma_domega_mumu(1.0, sqrt_s=10.0)), 6)
    0.5
    """
    cos_theta = np.asarray(cos_theta, dtype=float)
    s = sqrt_s**2
    return (alpha**2 / (4.0 * s)) * (1.0 + cos_theta**2)


def qed_total_cross_section_mumu(sqrt_s, alpha=ALPHA_FS):
    r"""Total leading-order QED cross section for :math:`e^+e^-\to\mu^+\mu^-`.

    .. math::

        \sigma = \frac{4\pi\alpha^2}{3s}

    obtained by integrating :func:`qed_dsigma_domega_mumu` over the full
    solid angle; the same simplifications apply (massless fermions, no
    :math:`Z` contribution).

    Parameters
    ----------
    sqrt_s : float
        Center-of-mass energy.
    alpha : float, default=ALPHA_FS
        Fine-structure constant.

    Returns
    -------
    float

    Examples
    --------
    >>> round(qed_total_cross_section_mumu(10.0), 10) == round(4 * np.pi * ALPHA_FS**2 / (3 * 100.0), 10)
    True
    """
    s = sqrt_s**2
    return 4.0 * np.pi * alpha**2 / (3.0 * s)


def meson_decay_rates_cp_eigenstate(t, delta_m, gamma_s, gamma_l, epsilon):
    r"""Time-dependent :math:`P^0,\bar P^0\to f_{CP}` decay rates, Wigner-Weisskopf toy model.

    A neutral meson :math:`P^0` (modeled on the :math:`K^0`-:math:`\bar
    K^0` system) mixes through the effective non-Hermitian Hamiltonian
    :math:`H=M-i\Gamma/2`; its mass eigenstates are the short/long-lived
    combinations :math:`P_{S,L}\approx P^0\pm\bar P^0` (up to the small
    CP-violating admixture parametrized by the complex parameter
    :math:`\varepsilon`, :math:`|\varepsilon|\ll1`). Decaying to a common
    CP eigenstate final state :math:`f` (e.g. :math:`\pi^+\pi^-`), the
    two flavor-tagged rates interfere as

    .. math::

        \Gamma(P^0(t)\to f) &= e^{-\Gamma_S t} + |\varepsilon|^2e^{-\Gamma_L t}
        + 2|\varepsilon|e^{-\Gamma t}\cos(\Delta m\,t-\phi_\varepsilon) \\
        \Gamma(\bar P^0(t)\to f) &= e^{-\Gamma_S t} + |\varepsilon|^2e^{-\Gamma_L t}
        - 2|\varepsilon|e^{-\Gamma t}\cos(\Delta m\,t-\phi_\varepsilon)

    with :math:`\Gamma=(\Gamma_S+\Gamma_L)/2`; this is the standard
    interference pattern used to measure :math:`\Delta m` and the
    :math:`K_S` lifetime from :math:`K^0/\bar K^0\to\pi\pi` decay-rate
    oscillations (see the Particle Data Group's "CP violation in
    :math:`K_L` decays" review, or Griffiths, *Introduction to Elementary
    Particles*, Ch. 8).

    Parameters
    ----------
    t : array_like
        Proper time since production (tagged as a pure :math:`P^0` or
        :math:`\bar P^0` flavor eigenstate at :math:`t=0`).
    delta_m : float
        Mass splitting between the two mass eigenstates, :math:`\Delta m=m_L-m_S`.
    gamma_s, gamma_l : float
        Decay widths of the short- and long-lived mass eigenstates.
    epsilon : complex
        CP-violation parameter, :math:`|\varepsilon|\ll1`.

    Returns
    -------
    gamma_meson, gamma_mesonbar : ndarray

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 5, 6)
    >>> g, gbar = meson_decay_rates_cp_eigenstate(t, delta_m=1.0, gamma_s=5.0, gamma_l=0.5, epsilon=0.0)
    >>> np.allclose(g, gbar)
    True
    """
    t = np.asarray(t, dtype=float)
    eps = complex(epsilon)
    eps_abs, phi_eps = abs(eps), np.angle(eps)
    gamma = 0.5 * (gamma_s + gamma_l)
    common = np.exp(-gamma_s * t) + eps_abs**2 * np.exp(-gamma_l * t)
    osc = 2.0 * eps_abs * np.exp(-gamma * t) * np.cos(delta_m * t - phi_eps)
    return common + osc, common - osc


def cp_asymmetry(t, delta_m, gamma_s, gamma_l, epsilon):
    r"""Decay-rate CP asymmetry :math:`A(t)`, from :func:`meson_decay_rates_cp_eigenstate`.

    .. math::

        A(t) = \frac{\Gamma(\bar P^0(t)\to f) - \Gamma(P^0(t)\to f)}
        {\Gamma(\bar P^0(t)\to f) + \Gamma(P^0(t)\to f)}

    which oscillates at frequency :math:`\Delta m` inside a decaying
    envelope, vanishing identically for :math:`\varepsilon=0` (no CP
    violation).

    Parameters
    ----------
    t, delta_m, gamma_s, gamma_l, epsilon
        See :func:`meson_decay_rates_cp_eigenstate`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 5, 6)
    >>> A = cp_asymmetry(t, delta_m=1.0, gamma_s=5.0, gamma_l=0.5, epsilon=0.0)
    >>> np.allclose(A, 0.0)
    True
    """
    gamma_meson, gamma_mesonbar = meson_decay_rates_cp_eigenstate(t, delta_m, gamma_s, gamma_l, epsilon)
    return (gamma_mesonbar - gamma_meson) / (gamma_mesonbar + gamma_meson)
