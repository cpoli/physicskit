"""Tracy-Widom soft-edge benchmark for the Gaussian ensembles.

References: C. A. Tracy, H. Widom, Commun. Math. Phys. 159 (1994) 151;
Commun. Math. Phys. 177 (1996) 727.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from scipy.stats import kstest, wasserstein_distance

from ..ensembles.base import MatrixEnsemble
from ..spectrum import Spectrum
from ..stats.tracy_widom import (
    largest_eigenvalues,
    tracy_widom_cdf,
    tracy_widom_edge_scale,
    tracy_widom_rvs,
)
from .base import ValidationResult


class TracyWidom:
    """Validates the largest-eigenvalue statistic of a Gaussian-ensemble
    Spectrum against the Tracy-Widom soft-edge law at the matching beta.

    Only beta in {1, 2, 4} are supported (F_1, F_2, F_4 have closed
    forms; general beta does not -- see ``physicskit.rmt.stats.tracy_widom``).
    Overrides ``validate`` (rather than just providing
    ``theoretical_cdf``/``reference_samples``) because the relevant
    statistic is the *largest* eigenvalue after a beta-dependent edge
    rescaling -- not the full ``spectrum.rescaled`` array the base class
    compares directly.
    """

    reference_size: int = 200_000

    def __init__(self, beta: int) -> None:
        if beta not in (1, 2, 4):
            raise ValueError(f"TracyWidom only supports beta in (1, 2, 4), got {beta}")
        self.beta = beta

    def theoretical_cdf(self, s: np.ndarray) -> np.ndarray:
        return tracy_widom_cdf(s, self.beta)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        return tracy_widom_rvs(size, self.beta, rng)

    def validate(self, spectrum: Spectrum, seed: int | np.random.Generator | None = None) -> ValidationResult:
        if spectrum.beta != self.beta:
            raise ValueError(f"Spectrum beta ({spectrum.beta}) does not match TracyWidom benchmark beta ({self.beta})")
        lam_max = largest_eigenvalues(spectrum)
        scale = tracy_widom_edge_scale(spectrum.n, self.beta)
        edge_stat = scale * (lam_max - 2.0)

        ks = kstest(edge_stat, self.theoretical_cdf)
        rng = np.random.default_rng(seed)
        reference = self.reference_samples(self.reference_size, rng)
        wd = wasserstein_distance(edge_stat, reference)
        return ValidationResult(
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wd),
            n_eigenvalues=len(edge_stat),
        )

    def convergence_curve(
        self,
        ensemble_factory: Callable[..., MatrixEnsemble],
        n_values: list[int],
        n_samples: int = 200,
        seed: int | np.random.Generator | None = None,
    ) -> list[tuple[int, ValidationResult]]:
        """Convergence to Tracy-Widom is notoriously slow (the largest
        eigenvalue is a single extreme-value statistic per sample, so
        each matrix contributes only one data point -- n_samples needs
        to be much larger than in the bulk-statistic benchmarks to get
        a stable KS estimate at all). Expect a shallower, noisier trend
        than the semicircle/Marchenko-Pastur convergence curves."""
        rng = np.random.default_rng(seed)
        results = []
        for n in n_values:
            ensemble = ensemble_factory(n, seed=int(rng.integers(1 << 31)))
            spectrum = ensemble.sample(n_samples=n_samples)
            result = self.validate(spectrum, seed=int(rng.integers(1 << 31)))
            results.append((n, result))
        return results
