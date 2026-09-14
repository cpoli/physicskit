"""Benchmark for the consecutive-spacing ratio distribution.

Reference: Y. Y. Atas, E. Bogomolny, O. Giraud, G. Roux, "Distribution of
the Ratio of Consecutive Level Spacings in Random Matrix Ensembles",
Phys. Rev. Lett. 110, 084101 (2013).

Deliberately structured as an independent check from ``WignerSurmise``:
the ratio statistic needs no unfolding, so agreement here rules out
unfolding artifacts as the explanation for agreement with the
(unfolding-dependent) spacing distribution.
"""

import numpy as np
from scipy.stats import kstest, wasserstein_distance

from ..stats.ratios import RatioSurmise as _RatioSurmiseModel
from .base import ValidationResult


class RatioDistribution:
    """Validates a ratio-statistic sample against the Atas et al. (2013)
    surmise at a given Dyson index beta."""

    reference_size: int = 200_000

    def __init__(self, beta: float) -> None:
        self.beta = beta
        self._model = _RatioSurmiseModel(beta)

    def theoretical_pdf(self, r: np.ndarray) -> np.ndarray:
        return self._model.pdf(r)

    def theoretical_cdf(self, r: np.ndarray) -> np.ndarray:
        return self._model.cdf(r)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return self._model.rvs(size, rng)

    def validate(self, ratios: np.ndarray, seed: int | np.random.Generator | None = None) -> ValidationResult:
        """Compare an array of ratio statistics (each in [0, 1], from
        ``physicskit.rmt.stats.ratios.ratio_statistics``) against the surmise."""
        ks = kstest(ratios, self.theoretical_cdf)
        rng = np.random.default_rng(seed)
        reference = self.reference_samples(self.reference_size, rng)
        wd = wasserstein_distance(ratios, reference)
        return ValidationResult(
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wd),
            n_eigenvalues=len(ratios),
        )
