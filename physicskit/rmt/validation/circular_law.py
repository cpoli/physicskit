"""Circular law benchmark for the Ginibre ensembles.

Reference: J. Ginibre, J. Math. Phys. 6 (1965) 440.
"""

from collections.abc import Callable

import numpy as np

from ..ensembles.base import MatrixEnsemble
from ..spectrum import Spectrum
from ..stats.circular_law import circular_law_radial_cdf, circular_law_radial_pdf
from .base import Benchmark, ValidationResult


class CircularLaw(Benchmark):
    """Validates a Ginibre-type Spectrum's eigenvalue *radii* against the
    circular law's radial marginal (F(r) = r^2 on [0, 1]).

    Overrides ``validate`` (rather than just ``theoretical_cdf`` /
    ``reference_samples``) because the base class compares
    ``spectrum.rescaled`` directly, but here it's the eigenvalue
    *magnitudes* -- not the (complex) rescaled eigenvalues themselves --
    that are compared to a 1-D theoretical distribution.
    """

    def theoretical_pdf(self, r: np.ndarray) -> np.ndarray:
        return circular_law_radial_pdf(r)

    def theoretical_cdf(self, r: np.ndarray) -> np.ndarray:
        return circular_law_radial_cdf(r)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        # Inverse-CDF sampling: F(r) = r^2 => r = sqrt(U), U ~ Uniform(0,1)
        u = rng.uniform(0.0, 1.0, size=size)
        return np.sqrt(u)

    def validate(self, spectrum: Spectrum, seed: int | np.random.Generator | None = None) -> ValidationResult:
        radii = np.abs(spectrum.rescaled.ravel())
        ks_result = self._ks_and_wasserstein(radii, seed)
        return ks_result

    def _ks_and_wasserstein(self, radii: np.ndarray, seed: int | np.random.Generator | None) -> ValidationResult:
        from scipy.stats import kstest, wasserstein_distance

        ks = kstest(radii, self.theoretical_cdf)
        rng = np.random.default_rng(seed)
        reference = self.reference_samples(self.reference_size, rng)
        wd = wasserstein_distance(radii, reference)
        return ValidationResult(
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wd),
            n_eigenvalues=len(radii),
        )

    def convergence_curve(
        self,
        ensemble_factory: Callable[..., MatrixEnsemble],
        n_values: list[int],
        n_samples: int = 20,
        seed: int | np.random.Generator | None = None,
    ) -> list[tuple[int, ValidationResult]]:
        """Same convergence-rate philosophy as the other exact-limit
        benchmarks (semicircle, Marchenko-Pastur): track KS distance to
        the circular law as N grows. Convergence here is known to be
        slower than the Hermitian ensembles' (weaker eigenvalue rigidity
        for non-Hermitian matrices), so expect a shallower log-log slope
        -- verified during development to be reliably negative but
        smaller in magnitude than the semicircle/MP cases."""
        rng = np.random.default_rng(seed)
        results = []
        for n in n_values:
            ensemble = ensemble_factory(n, seed=int(rng.integers(1 << 31)))
            spectrum = ensemble.sample(n_samples=n_samples)
            result = self.validate(spectrum, seed=int(rng.integers(1 << 31)))
            results.append((n, result))
        return results
