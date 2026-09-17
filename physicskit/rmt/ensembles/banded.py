"""Power-law banded random matrices (PBRM) -- a random matrix model for
the Anderson localization transition, multifractality, and 1-D
disordered quantum chains with long-range hopping.

Reference: A. D. Mirlin, Y. V. Fyodorov, F.-M. Dittes, J. Quezada,
T. H. Seligman, "Transition from localized to extended eigenstates in
the ensemble of power-law random banded matrices", Phys. Rev. E 54
(1996) 3221.

Construction: a real symmetric n x n matrix H with independent Gaussian
entries, mean zero, whose variance decays as a power law in the
distance from the diagonal:

    Var(H_ij) = [1 + (``abs(i - j)`` / b)**(2*alpha)]**(-1)     (i != j)

with band-width parameter ``b`` and decay exponent ``alpha``. Two
limits recover familiar physics: alpha -> infinity (or b -> infinity)
recovers a strictly banded matrix (short-range hopping, Anderson-
localized eigenstates for a fixed finite band); alpha -> 0 recovers the
standard (unbanded) GOE (delocalized, semicircle statistics). The
celebrated special case is alpha = 1 (the default here): the model is
then exactly at a MULTIFRACTAL critical point, with a b-dependent
family of critical level statistics interpolating continuously between
Poisson (b -> 0) and Wigner-Dyson/GOE (b -> infinity) -- unlike every
other ensemble in this package, PBRM has no single universal limiting
bulk density or fixed Dyson-class level statistic; the level statistics
themselves are the tunable object of study, which is why no
``natural_scale`` rescaling beyond leaving raw energies alone is
attempted here (see ``tests/test_banded.py`` for what is checked
instead: the two limiting regimes' qualitative behavior, not a single
closed-form target).

Diagonal entries are independent standard Gaussian (order 1, like the
off-diagonal entries nearest the diagonal) -- their exact distribution
does not affect the localization physics, which is governed entirely by
the off-diagonal decay profile.
"""

from __future__ import annotations

import numpy as np

from .base import MatrixEnsemble


def _power_law_variance_profile(n: int, b: float, alpha: float) -> np.ndarray:
    idx = np.arange(n)
    distance = np.abs(idx[:, None] - idx[None, :])
    with np.errstate(divide="ignore"):
        variance = 1.0 / (1.0 + (distance / b) ** (2.0 * alpha))
    return variance


class PowerLawBandedEnsemble(MatrixEnsemble):
    """Power-law banded random matrix ensemble (see module docstring).

    Parameters
    ----------
    n : int
    b : float
        Band-width parameter (b > 0).
    alpha : float, optional
        Power-law decay exponent (default 1.0, the multifractal
        critical point).
    """

    beta: float = 1

    def __init__(
        self,
        n: int,
        b: float,
        alpha: float = 1.0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if b <= 0:
            raise ValueError(f"b must be positive, got {b}")
        super().__init__(n, seed=seed)
        self.b = float(b)
        self.alpha = float(alpha)

    def _sample_matrix(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        variance = _power_law_variance_profile(n, self.b, self.alpha)
        std = np.sqrt(variance)
        # A single Gaussian draw per (i, j) pair (i < j), mirrored to
        # (j, i) -- unlike GOE's (x + x.T)/sqrt(2), which averages TWO
        # independent draws and so needs an extra /sqrt(2), there is
        # only one random variable per off-diagonal pair here, so
        # Var(H_ij) = std_ij**2 = variance_ij exactly, with no further
        # factor -- confirmed numerically during development (see
        # tests/test_banded.py) after an initial /2 factor undershot
        # the target profile by half.
        upper = np.triu(rng.standard_normal((n, n)) * std, k=1)
        h = upper + upper.T
        h += np.diag(rng.standard_normal(n))
        return h

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return np.linalg.eigvalsh(self._sample_matrix(rng))

    def _sample_eigenvalues_and_vectors(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        # Eigenvector localization (inverse participation ratio,
        # multifractal spectrum) is the entire physical motivation for
        # this ensemble -- see physicskit.rmt.stats.localization and
        # tests/test_localization.py for the Anderson-localization-
        # transition check this enables (IPR flat/O(1/n) for b large,
        # O(1) for b small, as the band-width parameter tunes the
        # ensemble between GOE-like and strictly-banded-localized).
        eigenvalues, eigenvectors = np.linalg.eigh(self._sample_matrix(rng))
        return eigenvalues, eigenvectors

    def natural_scale(self) -> float:
        return 1.0
