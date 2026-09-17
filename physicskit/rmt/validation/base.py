"""Base class for paper-replication benchmarks.

A single fixed-N Kolmogorov-Smirnov test with a p > 0.05 pass criterion is
*not* a reliable way to validate convergence to an asymptotic law: KS test
power grows with N, so real, expected finite-size deviations become more
likely to trigger rejection as N grows, not less. Instead, every Benchmark
here supports ``convergence_curve``, which tracks a distance-to-theory
metric across a sequence of N values and expects it to *shrink* as N grows
-- that is the statistically meaningful way to demonstrate asymptotic
convergence.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.stats import kstest, wasserstein_distance

from ..ensembles.base import MatrixEnsemble
from ..spectrum import Spectrum


@dataclass
class ValidationResult:
    """Result of comparing one Spectrum against a benchmark's theoretical
    distribution."""

    ks_statistic: float
    ks_pvalue: float
    wasserstein_distance: float
    n_eigenvalues: int

    def __repr__(self) -> str:
        return (
            f"ValidationResult(ks_statistic={self.ks_statistic:.5f}, "
            f"ks_pvalue={self.ks_pvalue:.3g}, "
            f"wasserstein_distance={self.wasserstein_distance:.5f}, "
            f"n_eigenvalues={self.n_eigenvalues})"
        )


class Benchmark(ABC):
    """Ties one theoretical distribution to a validation procedure.

    Subclasses implement ``theoretical_cdf`` (for the one-sample KS test,
    which needs no simulated reference sample) and ``reference_samples``
    (for the Wasserstein-distance comparison, which does).
    """

    @abstractmethod
    def theoretical_cdf(self, x: np.ndarray) -> np.ndarray:
        """Exact theoretical CDF, vectorized over x."""
        raise NotImplementedError  # pragma: no cover -- unreachable, see MatrixEnsemble._sample_eigenvalues

    @abstractmethod
    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        """Draw ``size`` i.i.d. samples from the theoretical distribution."""
        raise NotImplementedError  # pragma: no cover -- unreachable, see MatrixEnsemble._sample_eigenvalues

    #: Reference-sample size used for the Wasserstein comparison. Kept
    #: large and fixed (rather than tied to len(data)) so the reference
    #: sample's own Monte Carlo noise stays well below the eigenvalue-side
    #: signal even once the ensemble has converged tightly to theory --
    #: otherwise the noise floor from a same-size reference sample can
    #: swamp the (small, real) improvement between two already-close N
    #: values and make the distance non-monotonic by chance.
    reference_size: int = 200_000

    def validate(self, spectrum: Spectrum, seed: int | np.random.Generator | None = None) -> ValidationResult:
        """Compare a Spectrum's rescaled eigenvalues against theory.

        Returns a ValidationResult with both a one-sample KS statistic
        (against the exact theoretical CDF) and a Wasserstein distance
        (against simulated reference samples) -- report both rather than
        gating on a single p-value threshold.
        """
        data = spectrum.rescaled.ravel()
        ks = kstest(data, self.theoretical_cdf)
        rng = np.random.default_rng(seed)
        reference = self.reference_samples(self.reference_size, rng)
        wd = wasserstein_distance(data, reference)
        return ValidationResult(
            ks_statistic=float(ks.statistic),
            ks_pvalue=float(ks.pvalue),
            wasserstein_distance=float(wd),
            n_eigenvalues=len(data),
        )

    def convergence_curve(
        self,
        ensemble_factory: Callable[..., MatrixEnsemble],
        n_values: list[int],
        n_samples: int = 20,
        seed: int | np.random.Generator | None = None,
    ) -> list[tuple[int, ValidationResult]]:
        """Track distance-to-theory across increasing matrix size N.

        Parameters
        ----------
        ensemble_factory : callable
            ``ensemble_factory(n, seed=...)`` returning a fresh
            ``MatrixEnsemble`` instance of size n.
        n_values : sequence of int
            Matrix sizes to test, in increasing order.
        n_samples : int
            Independent matrix draws to pool at each N.
        seed : int or None

        Returns
        -------
        list of (n, ValidationResult)
            The KS statistic (and Wasserstein distance) in each result is
            expected to shrink as n increases; this is the check to make,
            not a single p > 0.05 threshold at one N.
        """
        rng = np.random.default_rng(seed)
        results = []
        for n in n_values:
            ensemble = ensemble_factory(n, seed=int(rng.integers(1 << 31)))
            spectrum = ensemble.sample(n_samples=n_samples)
            result = self.validate(spectrum, seed=int(rng.integers(1 << 31)))
            results.append((n, result))
        return results
