"""Nearest-neighbor spacing distribution P(s), and the (generalized)
Wigner surmise it is validated against.

References
----------
E. P. Wigner -- the original surmise, for GOE (beta=1).
M. L. Mehta, *Random Matrices* (3rd ed.), Academic Press, 2004 -- the
GUE/GSE surmises and their derivation from the 2x2 ensemble.

The beta-dependent closed form used here::

    P_beta(s) = a(beta) * s**beta * exp(-b(beta) * s**2)

with a, b fixed by normalization (integral P ds = 1) and unit mean
spacing (integral s*P ds = 1), reduces exactly to the classical
GOE/GUE/GSE surmises at beta = 1, 2, 4 and is the standard two-parameter
generalization to continuum beta (see e.g. G. Livan, M. Novaes, P. Vivo,
"Introduction to Random Matrices: Theory and Practice", Springer, 2018,
Sec. 2.4). It is a surmise -- exact for the 2x2 ensemble, an excellent
but not exact approximation to the true N-level spacing distribution --
which is why it is validated here as a benchmark rather than assumed.
"""

from collections.abc import Callable

import numpy as np
from scipy.special import gamma, gammainc, gammaincinv

from ..spectrum import Spectrum
from .unfolding import unfold_by_cdf


def nearest_neighbor_spacings(
    spectrum: Spectrum,
    cdf_func: Callable[[np.ndarray], np.ndarray],
    edge_trim: float = 0.1,
) -> np.ndarray:
    """Unfolded nearest-neighbor spacings, pooled across every sample in
    a Spectrum.

    Parameters
    ----------
    spectrum : physicskit.rmt.spectrum.Spectrum
    cdf_func : callable
        Theoretical CDF used for unfolding (e.g.
        ``physicskit.rmt.stats.density.semicircle_cdf`` for the Gaussian
        ensembles).
    edge_trim : float
        Fraction of eigenvalues discarded from each edge *after*
        unfolding, before taking spacings. The mean spectral density --
        and therefore unfolding accuracy -- is least reliable near the
        spectrum edges (soft-edge Tracy-Widom effects live exactly
        there), so edge eigenvalues bias the bulk spacing statistic if
        included. Trimming must happen after unfolding, not before:
        unfolding uses the total eigenvalue count N as the scale factor,
        so trimming first would unfold against the wrong N and bias the
        mean spacing away from 1.

    Returns
    -------
    numpy.ndarray
        Pooled unfolded spacings from all samples.
    """
    all_spacings = []
    for row in spectrum.rescaled:
        n = len(row)
        unfolded = unfold_by_cdf(row, cdf_func)
        lo = int(edge_trim * n)
        hi = n - lo
        if hi - lo < 2:
            continue
        all_spacings.append(np.diff(unfolded[lo:hi]))
    return np.concatenate(all_spacings)


def circular_spacings(spectrum: Spectrum) -> np.ndarray:
    """Unfolded nearest-neighbor spacings for a circular ensemble
    (COE/CUE/CSE), including the wraparound spacing between the largest
    and smallest phase.

    Unlike the Gaussian/Wishart ensembles, a circular ensemble has no
    spectrum edge (it lives on a circle) and no unfolding-accuracy
    concerns (``spectrum.rescaled`` is already exactly unfolded, see
    ``physicskit.rmt.ensembles.circular``), so every spacing is used -- no edge
    trimming, and one extra "wraparound" spacing per sample compared to
    the linear-spectrum case.
    """
    all_spacings = []
    for row in spectrum.rescaled:
        n = len(row)
        row_sorted = np.sort(row)
        spacings = np.diff(row_sorted)
        wraparound = (row_sorted[0] + n) - row_sorted[-1]
        all_spacings.append(np.concatenate([spacings, [wraparound]]))
    return np.concatenate(all_spacings)


def wigner_surmise_params(beta: float) -> tuple[float, float]:
    """(a, b) for P_beta(s) = a * s**beta * exp(-b * s**2), normalized to
    integral P(s) ds = 1 and integral s * P(s) ds = 1 (unit mean spacing).

    Derivation: writing the two normalization integrals in terms of
    Gamma functions and eliminating a gives
        b = [Gamma((beta+2)/2) / Gamma((beta+1)/2)]**2
        a = 2 * b**((beta+1)/2) / Gamma((beta+1)/2)
    Verified numerically against the classical closed forms at
    beta=1 (a=pi/2, b=pi/4), beta=2 (a=32/pi^2, b=4/pi), and
    beta=4 (a=2**18/(3**6 pi**3), b=64/(9 pi)).
    """
    b = (gamma((beta + 2) / 2) / gamma((beta + 1) / 2)) ** 2
    a = 2 * b ** ((beta + 1) / 2) / gamma((beta + 1) / 2)
    return a, b


def wigner_surmise_pdf(s: np.ndarray, beta: float) -> np.ndarray:
    """Generalized Wigner surmise density at spacing(s) s."""
    a, b = wigner_surmise_params(beta)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    mask = s >= 0
    out[mask] = a * s[mask] ** beta * np.exp(-b * s[mask] ** 2)
    return out


def wigner_surmise_cdf(s: np.ndarray, beta: float) -> np.ndarray:
    """Generalized Wigner surmise CDF.

    Closed form: substituting u = t^2 in the defining integral reduces it
    to the regularized lower incomplete gamma function,
        F(s) = P((beta+1)/2, b*s^2)
    (verified numerically against direct quadrature of the PDF).
    """
    _, b = wigner_surmise_params(beta)
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s)
    mask = s >= 0
    out[mask] = gammainc((beta + 1) / 2, b * s[mask] ** 2)
    return out


def wigner_surmise_samples(size: int | tuple[int, ...], beta: float, rng: np.random.Generator) -> np.ndarray:
    """Exact samples from the generalized Wigner surmise via inverse-CDF
    sampling (the CDF's closed form as a regularized incomplete gamma
    function has a matching closed-form inverse, ``gammaincinv``, so no
    rejection sampling is needed)."""
    _, b = wigner_surmise_params(beta)
    u = rng.uniform(0.0, 1.0, size=size)
    return np.sqrt(gammaincinv((beta + 1) / 2, u) / b)
