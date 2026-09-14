"""Benchmark for the (generalized) Wigner surmise -- the nearest-neighbor
spacing distribution of the Gaussian ensembles.

Unlike ``WignerSemicircle``, this benchmark validates a *derived*
statistic (unfolded spacings), so ``validate`` here expects pre-computed
spacings (from ``physicskit.rmt.stats.spacing.nearest_neighbor_spacings``), not a
raw Spectrum -- unfolding and edge-trimming choices belong to the caller,
not baked into the benchmark.
"""

import numpy as np
from scipy.stats import kstest, wasserstein_distance

from ..stats.spacing import (
    wigner_surmise_cdf,
    wigner_surmise_pdf,
    wigner_surmise_samples,
)
from .base import ValidationResult


class WignerSurmise:
    """Validates a spacing sample against the generalized Wigner surmise
    at a given Dyson index beta.

    References: Wigner's original surmise (beta=1); Mehta, *Random
    Matrices* (3rd ed.), 2004, for the beta=2, 4 cases.
    """

    reference_size: int = 200_000

    def __init__(self, beta: float) -> None:
        self.beta = beta

    def theoretical_pdf(self, s: np.ndarray) -> np.ndarray:
        return wigner_surmise_pdf(s, self.beta)

    def theoretical_cdf(self, s: np.ndarray) -> np.ndarray:
        return wigner_surmise_cdf(s, self.beta)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return wigner_surmise_samples(size, self.beta, rng)

    def validate(self, spacings: np.ndarray, seed: int | np.random.Generator | None = None) -> ValidationResult:
        """Compare an array of (already unfolded, already edge-trimmed)
        spacings against the surmise."""
        ks = kstest(spacings, self.theoretical_cdf)
        rng = np.random.default_rng(seed)
        reference = self.reference_samples(self.reference_size, rng)
        wd = wasserstein_distance(spacings, reference)
        return ValidationResult(
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wd),
            n_eigenvalues=len(spacings),
        )
