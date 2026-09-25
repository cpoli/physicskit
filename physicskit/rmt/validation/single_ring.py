"""Single ring theorem benchmark for bi-unitarily-invariant non-Hermitian
ensembles.

References: Feinberg-Zee (1997); Guionnet-Krishnapur-Zeitouni (2011) --
see ``physicskit.rmt.ensembles.single_ring`` for the full construction and
precision notes.
"""

from __future__ import annotations

import numpy as np

from ..spectrum import Spectrum
from ..stats.single_ring import annulus_radial_cdf, annulus_radial_pdf
from .base import Benchmark, ValidationResult


class SingleRingTheorem(Benchmark):
    """Validates a bi-unitarily-invariant non-Hermitian Spectrum's
    eigenvalue *radii* against a uniform-in-area annulus
    ``r_in <= |z| <= r_out``.

    The single ring theorem (Guionnet-Krishnapur-Zeitouni 2011) fixes the
    ring's radii for any such ensemble, but its radial profile is set by
    the S-transform of the singular-value law and is uniform only in
    special cases -- notably the induced-Ginibre/Wishart singular values of
    :class:`~physicskit.rmt.ensembles.single_ring.NonHermitianWishartEnsemble`
    (radii from ``single_ring_radii_wishart_theory``). For other
    singular-value laws, use this only as a check of the support.

    Like ``CircularLaw`` (its r_in=0 special case), overrides
    ``validate`` because the base class compares ``spectrum.rescaled``
    directly, but here it's the eigenvalue *magnitudes* that are
    compared to a 1-D theoretical distribution.

    Parameters
    ----------
    r_in : float
    r_out : float
        The theoretical ring radii (see
        ``physicskit.rmt.stats.single_ring.single_ring_radii`` /
        ``single_ring_radii_wishart_theory``).
    """

    def __init__(self, r_in: float, r_out: float) -> None:
        if not 0.0 <= r_in < r_out:
            raise ValueError(f"require 0 <= r_in < r_out, got r_in={r_in}, r_out={r_out}")
        self.r_in = r_in
        self.r_out = r_out

    def theoretical_pdf(self, r: np.ndarray) -> np.ndarray:
        return annulus_radial_pdf(r, self.r_in, self.r_out)

    def theoretical_cdf(self, r: np.ndarray) -> np.ndarray:
        return annulus_radial_cdf(r, self.r_in, self.r_out)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        # Inverse-CDF sampling: F(r) = (r^2-r_in^2)/(r_out^2-r_in^2)
        # => r = sqrt(r_in^2 + U*(r_out^2-r_in^2)), U ~ Uniform(0,1)
        u = rng.uniform(0.0, 1.0, size=size)
        return np.sqrt(self.r_in**2 + u * (self.r_out**2 - self.r_in**2))

    def validate(self, spectrum: Spectrum, seed: int | np.random.Generator | None = None) -> ValidationResult:
        from scipy.stats import kstest, wasserstein_distance

        radii = np.abs(spectrum.rescaled.ravel())
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
