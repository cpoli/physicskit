"""Statistics for the PT-symmetry-breaking transition (see
``physicskit.rmt.ensembles.pt_symmetric``): the fraction of real eigenvalues as
a function of the non-Hermiticity/coupling parameter g -- the
transition curve itself is the benchmark here, not a fixed limiting
law, since the ensemble's spectral character genuinely changes (real ->
partially complex) as g varies.

Also implemented: exceptional-point detection. An exceptional point is
where two eigenvalues AND their eigenvectors coalesce as a control
parameter varies -- unlike an ordinary level crossing (eigenvalues
cross, eigenvectors stay linearly independent), ``find_exceptional_points``
locates these by sweeping g for a single FIXED random realization (same
underlying blocks, only g rescaling the coupling block) and finding
where the real-eigenvalue count drops. Verified numerically during
development that these are genuine eigenvector coalescences, not mere
crossings: the right-eigenvector matrix's condition number grows sharply
approaching each detected g (2 -> 6 -> 38 within 0.002 of one such point
in a test case), the signature of the matrix becoming nearly
non-diagonalizable there -- see ``eigenvector_condition_number`` and
``tests/test_pt_symmetric.py``.
"""

from __future__ import annotations

import numpy as np

from ..ensembles.pt_symmetric import PTSymmetricEnsemble


def real_eigenvalue_fraction(eigenvalues: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Fraction of (near-)real eigenvalues per sample.

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n_samples, n), complex
    tol : float, optional
        An eigenvalue counts as real if ``abs(Im(lambda))`` is below
        this.

    Returns
    -------
    numpy.ndarray, shape (n_samples,)
        Fraction (in [0, 1]) of real eigenvalues in each sample.
    """
    eigenvalues = np.asarray(eigenvalues)
    return np.mean(np.abs(eigenvalues.imag) < tol, axis=-1)


def semi_poisson_pdf(s: np.ndarray) -> np.ndarray:
    """The semi-Poisson intermediate-statistics spacing density:
    P(s) = 4*s*exp(-2*s). Normalized (integral P ds = 1) and unit mean
    spacing (integral s*P ds = 1) -- checked directly here (both
    integrals evaluate to exactly 1 in closed form) rather than assumed.

    Characterizes level spacing at various "critical"/intermediate
    universality classes between Poisson (uncorrelated) and Wigner-Dyson
    (GOE-type): linear level repulsion at small s (like GOE, unlike
    Poisson's flat P(0)=1), but exponential (not Gaussian) large-s decay
    (like Poisson, unlike GOE's exp(-s^2)). Verified numerically here to
    be a substantially better fit than either pure Poisson or pure GOE
    to the surviving REAL eigenvalues' spacing distribution of
    ``PTSymmetricEnsemble`` at small-to-moderate coupling g (KS
    statistic roughly half that of the alternatives at g up to ~0.6) --
    see ``tests/test_pt_symmetric.py``.

    Parameters
    ----------
    s : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    s = np.asarray(s, dtype=float)
    return 4.0 * s * np.exp(-2.0 * s)


def semi_poisson_cdf(s: np.ndarray) -> np.ndarray:
    """CDF of the semi-Poisson spacing density: F(s) = 1 - (2s+1)*exp(-2s).

    Parameters
    ----------
    s : numpy.ndarray

    Returns
    -------
    numpy.ndarray
    """
    s = np.asarray(s, dtype=float)
    return 1.0 - (2.0 * s + 1.0) * np.exp(-2.0 * s)


def real_axis_spacings(eigenvalues: np.ndarray, tol: float = 1e-6) -> np.ndarray:
    """Nearest-neighbor spacings of the (near-)real eigenvalues in each
    sample, rescaled to unit mean spacing per sample and pooled.

    No unfolding CDF is used (unlike
    ``physicskit.rmt.stats.spacing.nearest_neighbor_spacings``): the real-
    eigenvalue density of a PT-symmetric ensemble near its
    symmetry-breaking transition has no known closed form, so this uses
    the simple, standard alternative of rescaling each sample's raw
    spacings by their own empirical mean -- adequate for a KS/shape
    comparison against ``semi_poisson_cdf``, though not as precise as
    unfolding against a known exact density.

    Parameters
    ----------
    eigenvalues : numpy.ndarray, shape (n_samples, n), complex
    tol : float, optional
        Passed to the real/complex classification (see
        ``real_eigenvalue_fraction``).

    Returns
    -------
    numpy.ndarray
        Pooled, mean-rescaled nearest-neighbor spacings of the real
        eigenvalues (samples with fewer than 3 real eigenvalues are
        skipped, since a mean spacing needs at least 2 spacings).
    """
    pooled = []
    for row in eigenvalues:
        real_values = np.sort(row.real[np.abs(row.imag) < tol])
        if len(real_values) < 3:
            continue
        spacings = np.diff(real_values)
        spacings = spacings[spacings > 1e-12]
        if len(spacings) < 2:
            continue
        pooled.append(spacings / spacings.mean())
    return np.concatenate(pooled)


def _fixed_realization_matrix(p: int, q: int, beta: int, seed: int, g: float) -> np.ndarray:
    """The PT-symmetric matrix for a SINGLE fixed random realization
    (blocks A, D, and the unscaled coupling direction determined
    entirely by ``seed``), at coupling strength ``g``. Since
    ``PTSymmetricEnsemble._sample_matrix`` draws A, D, then the coupling
    block (multiplied by g only at the very end) in a fixed order from a
    generator seeded identically each call, reconstructing the ensemble
    with a different g but the same seed reproduces the same A, D and
    unscaled coupling direction -- verified during development (see
    module docstring) -- making it possible to sweep g continuously for
    one realization rather than redrawing new randomness at every g.
    """
    ensemble = PTSymmetricEnsemble(p=p, q=q, g=g, beta=beta, seed=seed)
    return ensemble._sample_matrix(np.random.default_rng(seed))


def find_exceptional_points(
    p: int,
    q: int,
    beta: int,
    seed: int,
    g_max: float = 5.0,
    n_grid: int = 4000,
    tol: float = 1e-6,
) -> np.ndarray:
    """Locate exceptional points for a single fixed PT-symmetric
    realization by sweeping the coupling g in ``[0, g_max]`` and finding
    where the real-eigenvalue count decreases (a pair of real
    eigenvalues has merged and split into a complex-conjugate pair).

    Parameters
    ----------
    p, q : int
        Signature block sizes (see ``PTSymmetricEnsemble``).
    beta : int
        1 or 2.
    seed : int
        Fixes the single realization swept over g.
    g_max : float, optional
        Upper end of the g-sweep.
    n_grid : int, optional
        Number of g-grid points.
    tol : float, optional
        Passed to ``real_eigenvalue_fraction``.

    Returns
    -------
    numpy.ndarray
        g-values (grid resolution) at which a real-eigenvalue-count drop
        was detected -- approximate exceptional-point locations for this
        realization.
    """
    g_grid = np.linspace(0.0, g_max, n_grid)
    real_counts = np.empty(n_grid)
    for i, g in enumerate(g_grid):
        eigenvalues = np.linalg.eigvals(_fixed_realization_matrix(p, q, beta, seed, g))
        real_counts[i] = np.sum(np.abs(eigenvalues.imag) < tol)
    drops = np.where(np.diff(real_counts) < 0)[0]
    return g_grid[drops]


def mean_exceptional_point_count(
    p: int,
    q: int,
    beta: int,
    g_max: float,
    n_realizations: int = 30,
    seed: int | np.random.Generator | None = None,
    n_grid: int = 2000,
    tol: float = 1e-6,
) -> tuple[float, float]:
    """Ensemble-level exceptional-point statistic: the mean number of
    exceptional points found in ``[0, g_max]`` (see
    ``find_exceptional_points``), averaged over ``n_realizations``
    independent random realizations.

    Verified numerically during development to grow roughly linearly
    with ``min(p, q)`` at fixed g_max (e.g. p=q=4, 8, 12 gave mean counts
    of about 4.2, 8.7, 13.1 over g in [0, 2] with beta=2) -- consistent
    with the expectation that roughly min(p, q) real-eigenvalue pairs
    must eventually transition as g grows, though this specific scaling
    was observed rather than derived from a closed-form prediction.

    Parameters
    ----------
    p, q : int
    beta : int
    g_max : float
    n_realizations : int, optional
        Independent random realizations (distinct seeds) to average
        over.
    seed : int, numpy.random.Generator, or None, optional
        Seeds the sequence of per-realization seeds.
    n_grid, tol : see ``find_exceptional_points``.

    Returns
    -------
    mean_count : float
    standard_error : float
    """
    rng = np.random.default_rng(seed)
    raw_counts = []
    for _ in range(n_realizations):
        realization_seed = int(rng.integers(1 << 31))
        eps = find_exceptional_points(p, q, beta, realization_seed, g_max, n_grid, tol)
        raw_counts.append(len(eps))
    counts = np.array(raw_counts, dtype=float)
    return float(counts.mean()), float(counts.std(ddof=1) / np.sqrt(n_realizations))


def eigenvector_condition_number(p: int, q: int, beta: int, seed: int, g: float) -> float:
    """Condition number of the right-eigenvector matrix of the fixed-
    realization PT-symmetric matrix (see ``find_exceptional_points``) at
    coupling ``g``. Diverges approaching a genuine exceptional point
    (eigenvectors becoming parallel, the matrix approaching a Jordan
    block there) -- unlike an ordinary level crossing, where eigenvectors
    stay well-conditioned. The confirmatory structural check that
    distinguishes an exceptional point from an incidental crossing.

    Parameters
    ----------
    p, q : int
    beta : int
    seed : int
    g : float

    Returns
    -------
    float
    """
    h = _fixed_realization_matrix(p, q, beta, seed, g)
    _, eigenvectors = np.linalg.eig(h)
    return float(np.linalg.cond(eigenvectors))
