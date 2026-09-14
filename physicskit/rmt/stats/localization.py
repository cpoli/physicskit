"""Eigenvector localization statistics: the inverse participation ratio
(IPR) and its exact value for Haar-distributed (fully delocalized)
eigenvectors.

Unlike every other statistic in this package, these operate on
eigenVECTORS rather than eigenvalues, so they require a ``Spectrum``
sampled with ``ensemble.sample(return_eigenvectors=True)`` (see
``physicskit.rmt.ensembles.base.MatrixEnsemble.sample``) -- not every ensemble
supports this (dense diagonalization is required; the fast tridiagonal-
model ensembles' eigenvalues are exact in distribution but their raw
eigenvectors are not Haar-distributed, so eigenvector support is only
implemented where it's been verified to give genuinely Haar-distributed
(GOE/GUE) or physically meaningful (``PowerLawBandedEnsemble``,
Anderson localization) eigenvectors -- see each ensemble's docstring).

For a normalized eigenvector psi (``sum_i |psi_i|^2 = 1``), the inverse
participation ratio is

    ``IPR(psi) = sum_i |psi_i|^4``

IPR ~ O(1/n) for a fully delocalized (extended) state spread evenly over
all n sites, and IPR ~ O(1) (n-independent) for a localized state
concentrated on a handful of sites -- the standard order parameter for
the Anderson localization transition.

Exact theory for a Haar-distributed (rotationally/unitarily/
symplectically invariant, i.e. GOE/GUE/GSE-type) unit vector: writing
each component's squared magnitude as a "beta real dimensions per site"
weight, ``(|psi_1|^2, ..., |psi_n|^2)`` is exactly Dirichlet(beta/2, ...,
beta/2) distributed, giving the closed-form moment

    E[IPR] = (beta/2 + 1) / (n*beta/2 + 1)

(reduces to 3/(n+2) at beta=1, 2/(n+1) at beta=2, 3/(2n+1) at beta=4 --
verified numerically here against direct Haar-vector Monte Carlo at all
three beta, and independently against the Dirichlet-moment formula's own
derivation, before being trusted -- see ``tests/test_localization.py``).
"""

from collections.abc import Callable

import numpy as np

from ..ensembles.base import MatrixEnsemble


def inverse_participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """Inverse participation ratio of each eigenvector (column).

    Parameters
    ----------
    eigenvectors : numpy.ndarray, shape (..., m, n)
        Eigenvectors as columns (as returned in
        ``Spectrum.eigenvectors``); the leading ``...`` axes (e.g. a
        sample axis) are preserved.

    Returns
    -------
    numpy.ndarray, shape (..., n)
        IPR of each column, in (0, 1].
    """
    eigenvectors = np.asarray(eigenvectors)
    weights = np.abs(eigenvectors) ** 2
    weights = weights / np.sum(weights, axis=-2, keepdims=True)
    return np.sum(weights**2, axis=-2)


def inverse_participation_ratio_quaternionic(eigenvectors: np.ndarray) -> np.ndarray:
    """Inverse participation ratio for GSE-type (beta=4) eigenvectors,
    folding pairs of complex components into per-site quaternion
    weights first.

    GSE eigenvectors (as returned by
    ``physicskit.rmt.ensembles.gaussian.GSE.sample(return_eigenvectors=True)``)
    are naturally 2n-dimensional complex vectors representing n
    quaternionic "sites" (rows ``2i, 2i+1`` are one quaternion's two
    complex components) -- calling the plain
    ``inverse_participation_ratio`` on them directly would treat all 2n
    complex components as independent sites, which is not the physically
    meaningful per-site statistic and does not match
    ``ipr_theory(n, beta=4)``. This function folds pairs first:
    ``weight_i = |v[2i]|^2 + |v[2i+1]|^2``, matching the definition used
    to verify ``ipr_theory`` for beta=4 (see
    ``physicskit.rmt.ensembles.gaussian._dense_hermite_quaternion``).

    Parameters
    ----------
    eigenvectors : numpy.ndarray, shape (..., 2n, n)
        GSE-type eigenvectors as columns (as returned in
        ``Spectrum.eigenvectors`` for ``GSE``).

    Returns
    -------
    numpy.ndarray, shape (..., n)
        IPR of each column, computed over the n quaternionic sites.
    """
    eigenvectors = np.asarray(eigenvectors)
    weights = np.abs(eigenvectors[..., 0::2, :]) ** 2 + np.abs(eigenvectors[..., 1::2, :]) ** 2
    weights = weights / np.sum(weights, axis=-2, keepdims=True)
    return np.sum(weights**2, axis=-2)


def generalized_ipr(eigenvectors: np.ndarray, q: float) -> np.ndarray:
    """Generalized inverse participation ratio I_q = sum_i p_i^q, for
    normalized site weights ``p_i = |psi_i|^2``. Reduces to
    ``inverse_participation_ratio`` at q=2.

    The building block for multifractal analysis (see
    ``multifractal_dimension``): at a genuinely multifractal critical
    point (e.g. ``PowerLawBandedEnsemble`` at alpha=1), ``E[I_q]``
    scales as ``n**(-D_q*(q-1))`` for a nontrivial, q-dependent D_q,
    rather than the fully-delocalized ``D_q=1`` (``E[I_q] ~ n**-(q-1)``)
    or fully-localized ``D_q=0`` (``E[I_q]`` roughly n-independent).

    Parameters
    ----------
    eigenvectors : numpy.ndarray, shape (..., m, n)
    q : float
        Moment order (q != 1; q=1 is the special "information dimension"
        case, requiring a different limiting definition, not implemented
        here).

    Returns
    -------
    numpy.ndarray, shape (..., n)
    """
    if q == 1:
        raise ValueError("q=1 (information dimension) needs a different limiting definition (based on sum p_i*log(p_i)), not implemented here")
    eigenvectors = np.asarray(eigenvectors)
    weights = np.abs(eigenvectors) ** 2
    weights = weights / np.sum(weights, axis=-2, keepdims=True)
    return np.sum(weights**q, axis=-2)


def mass_exponent(
    ensemble_factory: Callable[..., MatrixEnsemble],
    n_values: list[int],
    q: float,
    n_samples: int = 10,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate the multifractal mass exponent tau(q) by fitting the
    scaling of ``E[generalized_ipr(., q)]`` with system size n:

        E[I_q](n) ~ n^(-tau(q))  =>  tau(q) = -slope

    for a log-log linear regression of mean I_q against n across
    ``n_values``. Unlike ``generalized_ipr``, q=1 is supported directly:
    I_1 = sum_i p_i = 1 identically (site weights are already
    normalized probabilities), so tau(1) = 0 exactly, for any ensemble,
    at any n -- no separate "information dimension" limiting definition
    is needed for tau itself (only for D_q = tau(q)/(q-1), which has a
    removable singularity at q=1).

    ``multifractal_dimension`` and ``singularity_spectrum`` are both
    built on this.

    Parameters
    ----------
    ensemble_factory : callable
        ``ensemble_factory(n, seed=...)`` returning a fresh
        ``MatrixEnsemble`` instance of size n that supports
        ``sample(return_eigenvectors=True)``.
    n_values : list of int
        System sizes to sample at, in increasing order.
    q : float
        Moment order.
    n_samples : int, optional
        Independent matrix draws to pool at each n.
    seed : int, numpy.random.Generator, or None, optional

    Returns
    -------
    tau_q : float
    r_squared : float
        Goodness of fit of the log-log linear regression (close to 1
        indicates clean power-law scaling over the given n_values).
    """
    from scipy.stats import linregress

    if q == 1.0:
        return 0.0, 1.0

    rng = np.random.default_rng(seed)
    means = []
    for n in n_values:
        ensemble = ensemble_factory(n, seed=int(rng.integers(1 << 31)))
        spectrum = ensemble.sample(n_samples=n_samples, return_eigenvectors=True)
        assert spectrum.eigenvectors is not None
        means.append(generalized_ipr(spectrum.eigenvectors, q).mean())
    fit = linregress(np.log(n_values), np.log(means))
    return -fit.slope, fit.rvalue**2


def multifractal_dimension(
    ensemble_factory: Callable[..., MatrixEnsemble],
    n_values: list[int],
    q: float,
    n_samples: int = 10,
    seed: int | np.random.Generator | None = None,
) -> tuple[float, float]:
    """Estimate the multifractal dimension D_q = tau(q) / (q-1) (see
    ``mass_exponent``). Verified during development against the two
    limiting cases on ``PowerLawBandedEnsemble``: D_2 ~ 1 (delocalized,
    large band-width b) and D_2 ~ 0 (localized, small b), both matching
    to within Monte Carlo noise.

    Parameters
    ----------
    ensemble_factory : callable
        ``ensemble_factory(n, seed=...)`` returning a fresh
        ``MatrixEnsemble`` instance of size n that supports
        ``sample(return_eigenvectors=True)``.
    n_values : list of int
        System sizes to sample at, in increasing order.
    q : float
        Moment order (q != 1; q=1 is the special "information dimension"
        case, requiring a different limiting definition, not implemented
        here).
    n_samples : int, optional
        Independent matrix draws to pool at each n.
    seed : int, numpy.random.Generator, or None, optional

    Returns
    -------
    d_q : float
    r_squared : float
        Goodness of fit of the log-log linear regression (close to 1
        indicates clean power-law scaling over the given n_values).
    """
    if q == 1:
        raise ValueError("q=1 (information dimension) needs a different limiting definition (based on sum p_i*log(p_i)), not implemented here")
    tau_q, r_squared = mass_exponent(ensemble_factory, n_values, q, n_samples, seed)
    return tau_q / (q - 1.0), r_squared


def singularity_spectrum(
    ensemble_factory: Callable[..., MatrixEnsemble],
    n_values: list[int],
    q_values: np.ndarray,
    n_samples: int = 10,
    seed: int | np.random.Generator | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Estimate the multifractal singularity spectrum f(alpha) via the
    Legendre transform of the mass exponent tau(q) (see
    ``mass_exponent``):

        alpha(q) = d(tau)/dq          f(alpha) = q*alpha(q) - tau(q)

    computed here by evaluating ``mass_exponent`` at each q in
    ``q_values`` and numerically differentiating (``numpy.gradient``)
    the resulting tau(q) curve -- exact given tau(q), but tau(q) itself
    is only as reliable as the underlying finite-size scaling fit,
    which needs a wide enough ``n_values`` range and enough statistics
    per point, especially at large |q| (dominated by rare, extreme
    eigenvector configurations that converge more slowly with n than
    the bulk of the distribution -- a well-known practical difficulty in
    multifractal numerics, not specific to this implementation).

    Numerically observed during development (``PowerLawBandedEnsemble``
    at its multifractal critical point, alpha=1): f(alpha) comes out
    cleanly positive and concave for q roughly in [0, 1.5], but goes
    negative for q gtrsim 2 at the system sizes/sample counts tractable
    here. This MAY be a genuine "negative dimension" branch (a
    documented phenomenon in multifractal analysis, associated with
    rare fluctuations whose probability decays as a negative power of
    system size) rather than an error, but was not independently
    confirmed against a literature reference value -- treat q gtrsim 2
    results as indicative rather than exact, and prefer a wide
    ``n_values`` range and large ``n_samples`` there.

    Parameters
    ----------
    ensemble_factory : callable
    n_values : list of int
    q_values : numpy.ndarray
        Moment orders to evaluate tau(q) at (need not exclude q=1).
    n_samples : int, optional
    seed : int, numpy.random.Generator, or None, optional

    Returns
    -------
    alpha : numpy.ndarray
    f_alpha : numpy.ndarray
    """
    q_values = np.asarray(q_values, dtype=float)
    rng = np.random.default_rng(seed)
    taus = np.empty_like(q_values)
    for i, q in enumerate(q_values):
        sub_seed = int(rng.integers(1 << 31))
        taus[i], _ = mass_exponent(ensemble_factory, n_values, q, n_samples, seed=sub_seed)
    alpha = np.gradient(taus, q_values)
    f_alpha = q_values * alpha - taus
    return alpha, f_alpha


def participation_ratio(eigenvectors: np.ndarray) -> np.ndarray:
    """Participation ratio PR = 1/IPR of each eigenvector (column): the
    effective number of sites a state is spread over -- O(n) for a fully
    delocalized state, O(1) for a localized one.

    Parameters
    ----------
    eigenvectors : numpy.ndarray, shape (..., m, n)

    Returns
    -------
    numpy.ndarray, shape (..., n)
    """
    return 1.0 / inverse_participation_ratio(eigenvectors)


def ipr_theory(n: int, beta: float) -> float:
    """Exact mean IPR of a Haar-distributed (fully delocalized) unit
    vector of dimension ``n`` at Dyson index ``beta``:

        E[IPR] = (beta/2 + 1) / (n*beta/2 + 1)

    The delocalized reference value that a localized ensemble's
    empirical IPR (see ``inverse_participation_ratio``) is contrasted
    against -- analogous to ``number_variance_poisson`` as the
    uncorrelated baseline for spectral rigidity.

    Parameters
    ----------
    n : int
        Vector dimension.
    beta : float
        Dyson index (1, 2, or 4 for GOE/GUE/GSE; any beta > 0 valid).

    Returns
    -------
    float
    """
    alpha = beta / 2.0
    return (alpha + 1.0) / (n * alpha + 1.0)
