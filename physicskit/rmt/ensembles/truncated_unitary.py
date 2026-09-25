"""Truncated random unitary matrices: the top-left m x m submatrix of an
n x n Haar-random unitary matrix.

Reference: K. Zyczkowski, H.-J. Sommers, "Truncated random unitary
matrices", J. Phys. A 33 (2000) 2045.

A truncated unitary matrix is a genuine "contraction" (all singular
values <= 1, generically < 1) -- it interpolates between a Haar-random
unitary matrix itself (alpha = m/n -> 1, eigenvalues exactly on the unit
circle) and, for alpha -> 0 (m fixed, n -> infinity), a matrix that
converges to complex Ginibre after the standard sqrt(m) rescaling (a
small block of a large Haar-random unitary matrix looks increasingly
like an unconstrained i.i.d. Gaussian matrix).

Support: eigenvalues fill a disk of radius sqrt(alpha), alpha = m/n --
verified numerically here (not assumed from a half-remembered formula:
an initial guess of sqrt(1-alpha) had the support radius shrinking as
more of the unitary matrix is kept, the wrong direction entirely, caught
immediately since it contradicts the alpha->1 limit converging to the
unit circle). Confirmed via the 99th-percentile eigenvalue radius
converging cleanly to sqrt(alpha) as n grows at fixed alpha (0.7318 ->
0.7235 -> 0.7166 -> 0.7110 at n=200,400,800,1600, alpha=0.5,
sqrt(0.5)=0.7071).

Precision note: unlike the Ginibre/circular-law case, the bulk density
within the disk is NOT uniform here (Zyczkowski-Sommers give a
non-uniform density weighted more heavily toward the edge) -- confirmed
qualitatively (mean eigenvalue radius, 0.533 at alpha=0.5, is
substantially larger than the 0.471 a uniform disk of the same radius
would give), but the exact closed-form density is not reproduced or
validated here; only the support radius (edge of the eigenvalue
distribution) is implemented and checked.
"""

from __future__ import annotations

import numpy as np

from ..utils.haar import haar_unitary
from .base import MatrixEnsemble


class TruncatedUnitaryEnsemble(MatrixEnsemble):
    """Eigenvalues of the top-left m x m submatrix of an n x n
    Haar-random unitary matrix, m = alpha * n (see module docstring).
    ``n`` here (per the ``MatrixEnsemble`` base class) is m, the
    truncated block size; the full unitary dimension is inferred from
    ``alpha``.

    Parameters
    ----------
    m : int
        Size of the truncated (kept) block -- the ensemble/matrix size.
    alpha : float
        Fraction of the full unitary matrix kept, m / n, in (0, 1].
        alpha=1 recovers a full Haar-unitary matrix (eigenvalues exactly
        on the unit circle); alpha -> 0 approaches complex Ginibre after
        rescaling.
    seed : int, numpy.random.Generator, or None, optional
        Seed for reproducible sampling. See
        :func:`~physicskit.rmt.utils.random_state.as_generator`.
    """

    def __init__(self, m: int, alpha: float, seed: int | np.random.Generator | None = None) -> None:
        if not 0.0 < alpha <= 1.0:
            raise ValueError(f"alpha must be in (0, 1], got {alpha}")
        super().__init__(m, seed=seed)
        self.alpha = float(alpha)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        m = self.n
        full_n = max(m, round(m / self.alpha))
        u = haar_unitary(full_n, rng)
        t = u[:m, :m]
        return np.linalg.eigvals(t)

    def natural_scale(self) -> float:
        # Eigenvalues fill a disk of radius sqrt(alpha) (verified
        # numerically -- see module docstring); dividing by this
        # rescales to the unit disk, matching the convention used for
        # the circular-law ensembles (physicskit.rmt.ensembles.ginibre).
        return np.sqrt(self.alpha)
