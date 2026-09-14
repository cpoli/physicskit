"""Real-Ginibre real-eigenvalue-count benchmark.

Like ``physicskit.rmt.validation.sine_kernel.SineKernel``, this doesn't fit the
KS/Wasserstein ``Benchmark`` interface: it compares a scalar expected
COUNT (of real eigenvalues), not a sample against a 1-D distribution. It
gets its own lightweight result type instead.

Reference: A. Edelman, E. Kostlan, S. Shub, J. Amer. Math. Soc. 7 (1994)
247. See ``physicskit.rmt.stats.real_ginibre`` for the precision caveat: the
theoretical value used here is the verified large-n asymptotic
expansion, not the exact finite-n Edelman-Kostlan-Shub closed form.
"""

from dataclasses import dataclass

from ..spectrum import Spectrum
from ..stats.real_ginibre import real_eigenvalue_count_asymptotic, real_eigenvalue_count_empirical


@dataclass
class RealEigenvalueCountResult:
    """Result of comparing a GinOE Spectrum's empirical mean real-
    eigenvalue count against the asymptotic theoretical value."""

    empirical_mean: float
    theoretical: float
    n: int

    @property
    def relative_error(self) -> float:
        return abs(self.empirical_mean - self.theoretical) / self.theoretical

    def __repr__(self) -> str:
        return (
            f"RealEigenvalueCountResult(empirical_mean={self.empirical_mean:.4f}, "
            f"theoretical={self.theoretical:.4f}, n={self.n}, "
            f"relative_error={self.relative_error:.4f})"
        )


class RealGinibreEigenvalueCount:
    """Validates a real-Ginibre (``GinOE``) Spectrum's mean number of
    real eigenvalues per sample against the asymptotic expected count
    sqrt(2n/pi) + 1/2 (see ``physicskit.rmt.stats.real_ginibre`` for the
    asymptotic-vs-exact precision caveat)."""

    def __init__(self, tol: float = 1e-8) -> None:
        """
        Parameters
        ----------
        tol : float, optional
            Passed to ``real_eigenvalue_count_empirical``: an eigenvalue
            counts as real if ``abs(Im(lambda))`` is below this.
        """
        self.tol = tol

    def validate(self, spectrum: Spectrum) -> RealEigenvalueCountResult:
        counts = real_eigenvalue_count_empirical(spectrum.eigenvalues, tol=self.tol)
        theoretical = real_eigenvalue_count_asymptotic(spectrum.n)
        return RealEigenvalueCountResult(empirical_mean=float(counts.mean()), theoretical=theoretical, n=spectrum.n)
