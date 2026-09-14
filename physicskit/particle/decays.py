r"""Two-body decay kinematics and radioactive decay chains.

Uses the same natural-unit convention (:math:`c=1`) as
:mod:`physicskit.particle.kinematics`.

- :func:`two_body_decay_momentum`, :func:`two_body_decay` -- the
  kinematics of a parent particle decaying to two daughters.
- :func:`decay_constant`, :func:`half_life`, :func:`radioactive_decay_number`,
  :func:`activity` -- the exponential radioactive decay law.
- :func:`bateman_decay_chain` -- the Bateman equations for a linear chain
  of successive radioactive decays.
- :func:`michel_spectrum`, :func:`sample_michel_electron_energies`,
  :func:`muon_decay_event` -- the three-body :math:`\mu^-\to e^-\bar\nu_\mu\nu_e`
  weak decay and its (unpolarized, massless-electron) Michel energy spectrum.
"""

from __future__ import annotations

import numpy as np

from physicskit.particle.kinematics import FourVector, boost_generic

__all__ = [
    "two_body_decay_momentum",
    "two_body_decay",
    "decay_constant",
    "half_life",
    "radioactive_decay_number",
    "activity",
    "bateman_decay_chain",
    "michel_spectrum",
    "sample_michel_electron_energies",
    "muon_decay_event",
]


def two_body_decay_momentum(M, m1, m2):
    r"""Daughter momentum magnitude in a two-body decay, in the parent's rest frame.

    .. math::

        p^* = \frac{\sqrt{\lambda(M^2,m_1^2,m_2^2)}}{2M}, \qquad
        \lambda(a,b,c) = a^2+b^2+c^2-2ab-2bc-2ca

    Parameters
    ----------
    M : float
        Parent mass.
    m1, m2 : float
        Daughter masses.

    Returns
    -------
    float

    Raises
    ------
    ValueError
        If ``M < m1 + m2`` (energetically forbidden decay).

    Examples
    --------
    >>> round(two_body_decay_momentum(1.0, 0.4, 0.4), 6)
    0.3
    """
    if M < m1 + m2:
        raise ValueError(f"M={M} must be >= m1+m2={m1 + m2} for the decay to be allowed.")
    a, b, c = M**2, m1**2, m2**2
    lam = a**2 + b**2 + c**2 - 2 * a * b - 2 * b * c - 2 * c * a
    return float(np.sqrt(max(lam, 0.0)) / (2.0 * M))


def two_body_decay(M, m1, m2, cos_theta, phi):
    """The two daughters' four-momenta in the parent's rest frame.

    Parameters
    ----------
    M : float
        Parent mass.
    m1, m2 : float
        Daughter masses.
    cos_theta : float
        Cosine of daughter 1's polar angle from the z-axis.
    phi : float
        Daughter 1's azimuthal angle.

    Returns
    -------
    p1, p2 : FourVector
        The two daughters, back-to-back in the parent's rest frame.

    Examples
    --------
    >>> p1, p2 = two_body_decay(1.0, 0.4, 0.4, cos_theta=1.0, phi=0.0)
    >>> round(p1.E, 6), round(p1.pz, 6)
    (0.5, 0.3)
    """
    p_star = two_body_decay_momentum(M, m1, m2)
    sin_theta = np.sqrt(max(1.0 - cos_theta**2, 0.0))
    px = p_star * sin_theta * np.cos(phi)
    py = p_star * sin_theta * np.sin(phi)
    pz = p_star * cos_theta
    E1 = np.sqrt(p_star**2 + m1**2)
    E2 = np.sqrt(p_star**2 + m2**2)
    p1 = FourVector(float(E1), float(px), float(py), float(pz))
    p2 = FourVector(float(E2), float(-px), float(-py), float(-pz))
    return p1, p2


def decay_constant(half_life_):
    r"""Decay constant from half-life, :math:`\lambda=\ln2/t_{1/2}`.

    Examples
    --------
    >>> round(decay_constant(np.log(2)), 6)
    1.0
    """
    return float(np.log(2.0) / half_life_)


def half_life(decay_constant_):
    r"""Half-life from decay constant, :math:`t_{1/2}=\ln2/\lambda`.

    Examples
    --------
    >>> round(half_life(1.0), 6)
    0.693147
    """
    return float(np.log(2.0) / decay_constant_)


def radioactive_decay_number(N0, decay_constant_, t):
    r"""Surviving population, :math:`N(t)=N_0e^{-\lambda t}`.

    Parameters
    ----------
    N0 : float
        Initial population.
    decay_constant_ : float
        Decay constant :math:`\lambda`.
    t : array_like
        Time(s), same units as :math:`1/\lambda`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> round(float(radioactive_decay_number(100.0, 1.0, 0.693147)), 4)
    50.0
    """
    return N0 * np.exp(-decay_constant_ * np.asarray(t))


def activity(N, decay_constant_):
    r"""Decay rate (activity), :math:`A=\lambda N`.

    Examples
    --------
    >>> activity(100.0, 0.01)
    1.0
    """
    return decay_constant_ * N


def bateman_decay_chain(N0, decay_constants, t):
    r"""Population of every species in a linear radioactive decay chain.

    Solves the Bateman equations for a chain :math:`1\to2\to\cdots\to n`
    with decay constants ``decay_constants`` (species 1 starts with
    population ``N0``, all daughters start at zero):

    .. math::

        N_k(t) = N_0\Big(\prod_{i=1}^{k-1}\lambda_i\Big)
        \sum_{i=1}^{k}\frac{e^{-\lambda_i t}}
        {\prod_{j=1,j\neq i}^{k}(\lambda_j-\lambda_i)}

    with :math:`N_1(t)=N_0e^{-\lambda_1 t}`. A zero decay constant for the
    final species represents a stable end product.

    Parameters
    ----------
    N0 : float
        Initial population of species 1.
    decay_constants : sequence of float
        Decay constant of each of the ``n`` species in the chain. Must be
        pairwise distinct among the species actually populated (the
        closed-form sum divides by ``lambda_j - lambda_i``).
    t : array_like
        Times at which to evaluate the populations.

    Returns
    -------
    ndarray of shape (n, len(t))
        Population of each species over time.

    Examples
    --------
    >>> t = np.array([0.0, 1.0])
    >>> N = bateman_decay_chain(100.0, [1.0, 2.0], t)
    >>> N.shape
    (2, 2)
    >>> round(float(N[0, 0]), 6)
    100.0
    """
    t = np.asarray(t, dtype=float)
    lam = np.asarray(decay_constants, dtype=float)
    n = len(lam)
    N = np.zeros((n, len(t)))
    for k in range(n):  # k is 0-indexed species k+1
        prefactor = N0 * np.prod(lam[:k]) if k > 0 else N0
        total = np.zeros_like(t)
        for i in range(k + 1):
            denom = 1.0
            for j in range(k + 1):
                if j != i:
                    denom *= lam[j] - lam[i]
            total += np.exp(-lam[i] * t) / denom
        N[k] = prefactor * total
    return N


def michel_spectrum(x):
    r"""The (unpolarized, massless-electron) Michel spectrum shape for muon decay.

    .. math::

        \frac{d\Gamma}{dx} = 2x^2(3-2x), \qquad x=\frac{2E_e}{m_\mu}\in[0,1]

    the standard leading-order result for :math:`\mu^-\to e^-\bar\nu_\mu\nu_e`
    from the V-A four-fermion weak interaction (Michel 1950; see e.g.
    Commins & Bucksbaum, *Weak Interactions of Leptons and Quarks*, Ch. 4,
    or the muon-decay chapter of any standard particle physics text),
    in the limit :math:`m_e/m_\mu\to0` and for an unpolarized muon (no
    Michel parameter :math:`\rho,\eta,\xi,\delta` dependence -- those
    parametrize corrections from polarization and a nonzero electron
    mass/anomalous couplings, all set to their V-A/Standard-Model,
    unpolarized values here). Normalized so that :math:`\int_0^1
    (d\Gamma/dx)\,dx = 1`.

    Parameters
    ----------
    x : array_like
        Scaled electron energy, :math:`x=2E_e/m_\mu\in[0,1]`.

    Returns
    -------
    ndarray or float

    Examples
    --------
    >>> import numpy as np
    >>> x = np.linspace(0, 1, 100001)
    >>> round(float(np.trapezoid(michel_spectrum(x), x)), 4)
    1.0
    """
    x = np.asarray(x, dtype=float)
    return 2.0 * x**2 * (3.0 - 2.0 * x)


def sample_michel_electron_energies(n, rng=None):
    r"""Sample ``n`` scaled electron energies :math:`x=2E_e/m_\mu` from the Michel spectrum.

    Rejection sampling against the shape's bound: :math:`x^2(3-2x)\le1`
    on :math:`[0,1]` (attained at :math:`x=1`), so a uniform proposal on
    :math:`[0,1]\times[0,1]` accepted below the (unnormalized) curve
    :math:`x^2(3-2x)` reproduces :func:`michel_spectrum`.

    Parameters
    ----------
    n : int
        Number of samples.
    rng : numpy.random.Generator, optional
        Random number generator; a fresh default one is used if omitted.

    Returns
    -------
    ndarray of shape (n,)

    Examples
    --------
    >>> rng = np.random.default_rng(0)
    >>> xs = sample_michel_electron_energies(20000, rng=rng)
    >>> bool(0.68 < xs.mean() < 0.72)  # theoretical mean <x> = 7/10
    True
    """
    rng = np.random.default_rng() if rng is None else rng
    samples = np.empty(n)
    filled = 0
    while filled < n:
        batch = max(2 * (n - filled), 16)
        x_cand = rng.uniform(0.0, 1.0, size=batch)
        y_cand = rng.uniform(0.0, 1.0, size=batch)
        accepted = x_cand[y_cand <= x_cand**2 * (3.0 - 2.0 * x_cand)]
        take = accepted[: n - filled]
        samples[filled : filled + len(take)] = take
        filled += len(take)
    return samples


def muon_decay_event(m_mu, rng=None):
    r"""Sample one :math:`\mu^-\to e^-+\bar\nu_\mu+\nu_e` decay in the muon rest frame.

    Toy-but-exact three-body kinematics built from two chained exact
    two-body decays (reusing :func:`two_body_decay`, so energy-momentum
    is conserved to machine precision at both vertices):

    1. The electron's energy is drawn from the Michel spectrum (massless
       electron approximation) via :func:`sample_michel_electron_energies`,
       fixing the invariant mass of the recoiling neutrino pair through
       :math:`m_{\nu\nu}^2=m_\mu^2-2m_\mu E_e` (exact for :math:`m_e=0`).
       :func:`two_body_decay` then gives the electron and the
       neutrino-pair "system" four-momenta, back to back in the muon
       rest frame, with an isotropically sampled decay angle (the
       standard unpolarized-muon assumption).
    2. The neutrino-pair system is split into its two (massless)
       neutrinos isotropically in *its own* rest frame via a second call
       to :func:`two_body_decay`, then boosted back to the muon rest
       frame with :func:`~physicskit.particle.kinematics.boost_generic`.

    Both neutrino masses are set to zero -- an excellent approximation
    given their sub-eV masses compared to :math:`m_\mu\sim100` MeV.

    Parameters
    ----------
    m_mu : float
        Muon mass (in whatever energy unit the caller uses consistently).
    rng : numpy.random.Generator, optional
        Random number generator; a fresh default one is used if omitted.

    Returns
    -------
    p_e, p_numu_bar, p_nue : FourVector
        The three decay products, in the muon rest frame.

    Examples
    --------
    >>> from physicskit.particle.kinematics import invariant_mass
    >>> rng = np.random.default_rng(1)
    >>> p_e, p_numu_bar, p_nue = muon_decay_event(105.658, rng=rng)
    >>> round(invariant_mass([p_e, p_numu_bar, p_nue]), 6)
    105.658
    """
    rng = np.random.default_rng() if rng is None else rng
    x = float(sample_michel_electron_energies(1, rng=rng)[0])
    E_e = 0.5 * x * m_mu
    m_nunu = float(np.sqrt(max(m_mu**2 - 2.0 * m_mu * E_e, 0.0)))
    cos1, phi1 = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)
    p_e, p_nunu = two_body_decay(m_mu, 0.0, m_nunu, cos1, phi1)
    cos2, phi2 = rng.uniform(-1.0, 1.0), rng.uniform(0.0, 2.0 * np.pi)
    p_numu_bar_rest, p_nue_rest = two_body_decay(m_nunu, 0.0, 0.0, cos2, phi2)
    beta_vec = p_nunu.p_vec / p_nunu.E
    p_numu_bar = boost_generic(p_numu_bar_rest, beta_vec)
    p_nue = boost_generic(p_nue_rest, beta_vec)
    return p_e, p_numu_bar, p_nue
