"""Wigner semicircle law benchmark.

References
----------
E. Wigner, "Characteristic Vectors of Bordered Matrices With Infinite
Dimensions", Ann. Math. 62 (1955) 548.
E. Wigner, "On the Distribution of the Roots of Certain Symmetric
Matrices", Ann. Math. 67 (1958) 325.
"""

import numpy as np
from scipy.stats import beta as beta_dist

from ..stats.density import semicircle_cdf, semicircle_pdf
from .base import Benchmark


class WignerSemicircle(Benchmark):
    """Validates the empirical spectral density of a Gaussian
    beta-ensemble (GOE/GUE/GSE, or general beta-Hermite) against the
    exact Wigner semicircle law.

    The semicircle distribution on [-radius, radius] is an affine
    transform of Beta(3/2, 3/2): if U ~ Beta(3/2, 3/2), then
    X = radius * (2U - 1) has semicircle density, which is how
    ``reference_samples`` draws from the theoretical law.
    """

    def __init__(self, radius: float = 2.0) -> None:
        self.radius = radius

    def theoretical_cdf(self, x: np.ndarray) -> np.ndarray:
        return semicircle_cdf(x, radius=self.radius)

    def theoretical_pdf(self, x: np.ndarray) -> np.ndarray:
        return semicircle_pdf(x, radius=self.radius)

    def reference_samples(self, size: int, rng: np.random.Generator) -> np.ndarray:
        u = beta_dist.rvs(1.5, 1.5, size=size, random_state=rng)
        return self.radius * (2.0 * u - 1.0)
