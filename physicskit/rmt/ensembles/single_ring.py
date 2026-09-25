"""The single ring theorem, and a non-Hermitian Wishart-type ensemble
built on it.

References
----------
J. Feinberg, A. Zee, "Non-Hermitian random matrix theory: Method of
Hermitian reduction", Nucl. Phys. B 504 (1997) 579 -- the original
single ring conjecture.
A. Guionnet, M. Krishnapur, O. Zeitouni, "The single ring theorem",
Ann. of Math. 174 (2011) 1189 -- the rigorous proof.

Setup: M = U * diag(s_1, ..., s_n) * V, with U, V independent
Haar-random unitary matrices and s_1, ..., s_n the (fixed, or random and
independent of U, V) singular values of M. M is "bi-unitarily
invariant" -- its distribution is unchanged under M -> W1 M W2 for any
unitary W1, W2 -- which is the general class of non-Hermitian ensembles
the single ring theorem applies to (it says nothing about ensembles,
like a product of two INDEPENDENT Ginibre matrices, whose two
constructing factors are not related by a shared bi-unitary-invariant
structure -- that is a different, separately-studied class of "product
ensembles").

As n -> infinity, M's eigenvalues fill the annulus ``r_in <= |z| <= r_out``
UNIFORMLY (in the sense of the limiting radial density -- not
necessarily an exactly flat 2-D density off the boundary at finite n,
though for large n the bulk approaches uniform on the annulus), with

    r_out = sqrt(<s^2>)          r_in = 1 / sqrt(<1/s^2>)

(<.> the mean over the n singular values / the limiting singular-value
distribution). r_in = 0 exactly recovers a filled disk (e.g. Ginibre:
verified numerically here that construction gives r_in ~ 0, consistent
with the already-implemented ``physicskit.rmt.stats.circular_law``); r_in > 0
is a genuine "ring" with a hole around the origin, only possible for a
non-Hermitian (or more precisely, non-normal) ensemble.

Non-Hermitian Wishart-type ensemble (``NonHermitianWishartEnsemble``):
this package does not reproduce Akemann et al.'s exact finite-n,
skew-orthogonal-Laguerre-kernel-based density for a specific tunable-
non-Hermiticity-parameter construction (that is genuinely specialized
machinery this package does not otherwise need, and is not implemented
here -- see design notes). What IS implemented, and rigorously
validated via the single ring theorem above (no skew-orthogonal kernel
required for that -- it is a two-moment formula), is a legitimate
non-Hermitian generalization of the Wishart ensemble in the same
spirit: singular values s_i = sqrt(squared-singular-value), the squared
singular values drawn EXACTLY from the beta-Laguerre (Wishart)
distribution at aspect ratio gamma = n/m < 1 (reusing
``physicskit.rmt.utils.tridiagonal.sample_laguerre_beta_eigenvalues``, already
validated elsewhere in this package against the Marchenko-Pastur law),
then scrambled by independent Haar U, V. Because the Marchenko-Pastur
distribution has the classical exact moments E[X] = 1 and
E[1/X] = 1/(1-gamma) (gamma < 1) -- verified numerically here against
direct Monte Carlo before being trusted, see
``tests/test_single_ring.py`` -- this ensemble's exact theoretical
ring radii are

    r_out = 1          r_in = sqrt(1 - gamma)

a genuine ring (r_in > 0) whenever gamma < 1, unlike the balanced
(gamma=1) Wishart case, which was also checked here to collapse to
r_in ~ 0 (a filled disk), consistent with Ginibre's ordinary circular
law.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..utils.haar import haar_unitary
from ..utils.tridiagonal import sample_laguerre_beta_eigenvalues
from .base import MatrixEnsemble

SingularValueSampler = Callable[[np.random.Generator, int], np.ndarray]


class SingleRingEnsemble(MatrixEnsemble):
    """General bi-unitarily-invariant non-Hermitian ensemble
    M = U * diag(s) * V (see module docstring), with a caller-supplied
    singular-value sampler.

    Parameters
    ----------
    n : int
        Matrix dimension.
    singular_value_sampler : callable
        ``singular_value_sampler(rng, n)`` returning ``n`` non-negative
        singular values.
    seed : int, numpy.random.Generator, or None, optional
        Seed for reproducible sampling. See
        :func:`~physicskit.rmt.utils.random_state.as_generator`.
    """

    def __init__(
        self,
        n: int,
        singular_value_sampler: SingularValueSampler,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, seed=seed)
        self.singular_value_sampler = singular_value_sampler

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        singular_values = self.singular_value_sampler(rng, n)
        u = haar_unitary(n, rng)
        v = haar_unitary(n, rng)
        m = (u * singular_values) @ v
        return np.linalg.eigvals(m)

    def natural_scale(self) -> float:
        # No universal rescaling: the ring radii depend entirely on the
        # caller-supplied singular value distribution's own scale.
        return 1.0


class NonHermitianWishartEnsemble(MatrixEnsemble):
    """Non-Hermitian Wishart-type ensemble: singular values drawn from
    the exact beta-Laguerre (Marchenko-Pastur) distribution at aspect
    ratio gamma = n/m < 1, scrambled by independent Haar U, V (see
    module docstring). Eigenvalues fill a genuine ring (annulus) in the
    complex plane, ``r_in = sqrt(1-gamma) <= |z| <= r_out = 1``, unlike the
    ordinary (Hermitian) Wishart ensemble's real, non-negative
    eigenvalues.

    Parameters
    ----------
    n : int
        Number of variables (matrix dimension).
    m : int
        Number of samples; must be > n (gamma = n/m < 1 strictly, so the
        ring has a nonzero hole).
    beta : float, optional
        Dyson index of the underlying Wishart-type singular-value
        distribution (default 2); any beta > 0 valid continuum value.
    seed : int, numpy.random.Generator, or None, optional
        Seed for reproducible sampling. See
        :func:`~physicskit.rmt.utils.random_state.as_generator`.
    """

    def __init__(
        self,
        n: int,
        m: int,
        beta: float = 2.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if m <= n:
            raise ValueError(f"m (samples={m}) must be > n (variables={n}) for gamma < 1")
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        super().__init__(n, seed=seed)
        self.m = m
        self.beta_wishart: float = float(beta)

    @property
    def gamma(self) -> float:
        """Aspect ratio n/m controlling the ring's inner radius."""
        return self.n / self.m

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        squared_singular_values = sample_laguerre_beta_eigenvalues(self.m, n, self.beta_wishart, rng)
        singular_values = np.sqrt(squared_singular_values)
        u = haar_unitary(n, rng)
        v = haar_unitary(n, rng)
        matrix = (u * singular_values) @ v
        return np.linalg.eigvals(matrix)

    def natural_scale(self) -> float:
        # The Marchenko-Pastur-derived singular values are already in
        # the natural r_out=1 normalization; no further rescaling.
        return 1.0
