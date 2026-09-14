r"""Anderson localization: the 1958 discovery that disorder can halt diffusion entirely.

Philip Anderson showed that a quantum particle hopping on a lattice with
sufficiently strong, random (quenched) onsite disorder does not diffuse at
all: interference between all the scattering paths off the random
potential can exponentially localize *every* eigenstate, at any disorder
strength, in one and two dimensions. This was the first demonstration that
disorder is not merely a small perturbative correction to a metal's
conductivity, but can drive a genuine transition -- absent in three
dimensions only above a critical disorder (the "mobility edge") -- from
extended, conducting states to localized, insulating ones. It underlies
the disorder-driven physics that makes real quantum Hall plateaus finite in
width (:mod:`physicskit.condensed.topology`) and the transport (or lack of
it) in every real, imperfect crystal.

Uses the same natural-unit convention (:math:`\hbar=e=1`) as the rest of
:mod:`physicskit.condensed`.
"""

from __future__ import annotations

import numpy as np

__all__ = ["anderson_chain_hamiltonian", "inverse_participation_ratio", "localization_length"]


def anderson_chain_hamiltonian(n_sites: int, disorder_strength: float, t: float = 1.0, seed: int | None = None) -> np.ndarray:
    r"""Real-space 1D Anderson model: a tight-binding chain with random onsite disorder.

    .. math::

       H = -t\sum_i \left(c_i^\dagger c_{i+1} + \text{h.c.}\right)
       + \sum_i \epsilon_i\, c_i^\dagger c_i, \qquad
       \epsilon_i \sim \text{Uniform}\!\left(-\tfrac{W}{2}, \tfrac{W}{2}\right),

    with open boundary conditions.

    Parameters
    ----------
    n_sites : int
        Number of lattice sites.
    disorder_strength : float
        Disorder width :math:`W`. ``0`` recovers the clean chain.
    t : float, default=1.0
        Nearest-neighbor hopping amplitude.
    seed : int, optional
        Seed for the random onsite disorder, for reproducibility.

    Returns
    -------
    ndarray, shape (n_sites, n_sites)
        Real, symmetric Hamiltonian matrix.

    Examples
    --------
    >>> H = anderson_chain_hamiltonian(n_sites=4, disorder_strength=0.0)
    >>> H
    array([[ 0., -1.,  0.,  0.],
           [-1.,  0., -1.,  0.],
           [ 0., -1.,  0., -1.],
           [ 0.,  0., -1.,  0.]])
    """
    rng = np.random.default_rng(seed)
    H = np.diag(rng.uniform(-disorder_strength / 2, disorder_strength / 2, size=n_sites))
    idx = np.arange(n_sites - 1)
    H[idx, idx + 1] = -t
    H[idx + 1, idx] = -t
    return H


def inverse_participation_ratio(psi) -> float:
    r"""Inverse participation ratio :math:`\text{IPR} = \sum_i|\psi_i|^4 / \left(\sum_i|\psi_i|^2\right)^2`.

    A dimensionless measure of how many sites an eigenstate is spread
    over: :math:`\text{IPR}\sim 1/N` for a state extended over all
    :math:`N` sites, and :math:`\text{IPR} = O(1)`, independent of
    :math:`N`, for a state localized on a handful of sites -- the
    diagnostic Anderson used to distinguish the two regimes.

    Parameters
    ----------
    psi : array_like
        Eigenvector components (need not be pre-normalized).

    Returns
    -------
    float

    Examples
    --------
    A state spread equally over ``N`` sites has IPR = 1/N:

    >>> import numpy as np
    >>> psi = np.ones(10) / np.sqrt(10)
    >>> round(inverse_participation_ratio(psi), 6)
    0.1
    >>> psi_localized = np.zeros(10); psi_localized[0] = 1.0
    >>> inverse_participation_ratio(psi_localized)
    1.0
    """
    psi = np.asarray(psi)
    p = np.abs(psi) ** 2
    return float(np.sum(p**2) / np.sum(p) ** 2)


def localization_length(psi, positions=None) -> float:
    r"""Estimate an eigenstate's localization length from the decay of its envelope.

    Fits :math:`\log|\psi_i|^2` linearly against position on either side of
    the state's peak and returns :math:`\xi = -2/\text{slope}`, averaged
    over both sides -- the length scale over which a localized state's
    probability density decays as :math:`|\psi(x)|^2 \sim e^{-2|x-x_0|/\xi}`.
    Returns ``inf`` for a state too extended (flat/non-monotonic envelope)
    for the fit to detect exponential decay.

    Parameters
    ----------
    psi : array_like
        Eigenvector components, indexed by site.
    positions : array_like, optional
        Site coordinates. Defaults to integer indices ``0, 1, ..., N-1``.

    Returns
    -------
    float

    Examples
    --------
    An exactly exponential envelope recovers its localization length:

    >>> import numpy as np
    >>> x = np.arange(200)
    >>> xi_true = 5.0
    >>> psi = np.exp(-np.abs(x - 100) / xi_true)
    >>> round(localization_length(psi), 4)
    5.0
    """
    psi = np.asarray(psi)
    positions = np.arange(len(psi)) if positions is None else np.asarray(positions, dtype=float)
    log_density = np.log(np.abs(psi) ** 2 + 1e-300)
    peak = int(np.argmax(log_density))

    slopes = []
    if peak > 1:
        left_slope = np.polyfit(positions[: peak + 1], log_density[: peak + 1], 1)[0]
        if left_slope > 0:
            slopes.append(left_slope)
    if peak < len(psi) - 2:
        right_slope = np.polyfit(positions[peak:], log_density[peak:], 1)[0]
        if right_slope < 0:
            slopes.append(-right_slope)

    if not slopes:
        return float("inf")
    return float(2.0 / np.mean(slopes))
