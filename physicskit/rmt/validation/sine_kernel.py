"""Sine-kernel benchmark for the Circular Unitary Ensemble's bulk
two-point correlation function.

This does not fit the KS/Wasserstein ``Benchmark`` interface used
elsewhere in the package: it validates a *correlation function* (a
density estimated by pair-counting), not a sample from a 1-D
distribution. It gets its own lightweight result type instead.

Reference: F. J. Dyson, J. Math. Phys. 3 (1962) 140, 157, 166.
"""

from dataclasses import dataclass

import numpy as np

from ..spectrum import Spectrum
from ..stats.correlations import pair_correlation_estimate, sine_kernel_r2


@dataclass
class CorrelationValidationResult:
    rmse: float
    r_max: float
    n_bins: int

    def __repr__(self) -> str:
        return f"CorrelationValidationResult(rmse={self.rmse:.5f}, r_max={self.r_max}, n_bins={self.n_bins})"


class SineKernel:
    """Validates a CUE (beta=2) Spectrum's bulk two-point correlation
    function against the exact determinantal sine-kernel result.

    Only beta=2 is implemented -- see module docstring in
    ``physicskit.rmt.stats.correlations`` for why the beta=1, 4 cases (COE, CSE)
    are deferred rather than guessed at.
    """

    def __init__(self, r_max: float = 4.0, n_bins: int = 60) -> None:
        self.r_max = r_max
        self.n_bins = n_bins

    def theoretical_r2(self, r: np.ndarray) -> np.ndarray:
        return sine_kernel_r2(r)

    def validate(self, spectrum: Spectrum) -> CorrelationValidationResult:
        if spectrum.beta != 2:
            raise ValueError(
                "SineKernel currently only validates beta=2 (CUE) spectra; "
                "the beta=1 (COE) and beta=4 (CSE) correlation kernels "
                "require the fuller Pfaffian point-process machinery, "
                "not yet implemented (see physicskit.rmt.stats.correlations)."
            )
        centers, estimate = pair_correlation_estimate(spectrum, r_max=self.r_max, n_bins=self.n_bins)
        theory = self.theoretical_r2(centers)
        rmse = float(np.sqrt(np.mean((estimate - theory) ** 2)))
        return CorrelationValidationResult(rmse=rmse, r_max=self.r_max, n_bins=self.n_bins)
