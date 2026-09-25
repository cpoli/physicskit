"""Wishart (Laguerre) ensembles -- LOE, LUE, LSE -- plus the continuum
Dyson-index generalization, built on the bidiagonal beta-Laguerre model
(see ``physicskit.rmt.utils.tridiagonal.sample_laguerre_beta_eigenvalues``).

References
----------
V. A. Marchenko, L. A. Pastur, "Distribution of eigenvalues for some sets
of random matrices", Mat. Sb. 72 (1967) 507.
I. Dumitriu, A. Edelman, J. Math. Phys. 43, 5830 (2002).
"""

from __future__ import annotations

import numpy as np

from ..utils.tridiagonal import sample_laguerre_beta_eigenvalues
from .base import MatrixEnsemble


class LaguerreBetaEnsemble(MatrixEnsemble):
    """General beta-Laguerre (Wishart) ensemble.

    Corresponds to an m x n rectangular data matrix (m "samples", n
    "variables", m >= n) with i.i.d. unit-variance entries. beta=1, 2, 4
    recover LOE, LUE, LSE exactly; any beta > 0 is a valid continuum
    generalization.

    ``n`` (the ensemble/matrix size, per the ``MatrixEnsemble`` base
    class) is the number of variables; ``m`` is the number of samples,
    with aspect ratio gamma = n / m controlling the Marchenko-Pastur
    support.
    """

    def __init__(self, n: int, m: int, beta: float, seed: int | np.random.Generator | None = None) -> None:
        """
        Parameters
        ----------
        n : int
            Number of variables (matrix dimension).
        m : int
            Number of samples; must be >= n.
        beta : float
            Dyson index; must be positive.
        seed : int, numpy.random.Generator, or None, optional
            Seed for reproducible sampling. See
            :func:`~physicskit.rmt.utils.random_state.as_generator`.
        """
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if m < n:
            raise ValueError(f"m (samples={m}) must be >= n (variables={n})")
        super().__init__(n, seed=seed)
        self.m = m
        self.beta: float = float(beta)

    @property
    def gamma(self) -> float:
        """Aspect ratio n/m controlling the Marchenko-Pastur support."""
        return self.n / self.m

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return sample_laguerre_beta_eigenvalues(self.m, self.n, self.beta, rng)

    def natural_scale(self) -> float:
        # The bidiagonal construction already returns eigenvalues in the
        # standard Marchenko-Pastur normalization (support
        # [(1-sqrt(gamma))^2, (1+sqrt(gamma))^2]); no further rescaling
        # needed.
        return 1.0


class LOE(LaguerreBetaEnsemble):
    """Laguerre Orthogonal Ensemble (beta=1): real Wishart matrices."""

    def __init__(self, n: int, m: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m, beta=1, seed=seed)


class LUE(LaguerreBetaEnsemble):
    """Laguerre Unitary Ensemble (beta=2): complex Wishart matrices."""

    def __init__(self, n: int, m: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m, beta=2, seed=seed)


class LSE(LaguerreBetaEnsemble):
    """Laguerre Symplectic Ensemble (beta=4): quaternionic Wishart matrices."""

    def __init__(self, n: int, m: int, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, m, beta=4, seed=seed)
