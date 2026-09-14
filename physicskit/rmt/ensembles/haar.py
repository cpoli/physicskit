"""Eigenvalue statistics of a raw Haar-random orthogonal matrix --
deliberately distinct from the Circular Orthogonal Ensemble.

Reference: F. Mezzadri, "How to generate random matrices from the
classical compact groups", Notices Amer. Math. Soc. 54 (2007) 592.

As already flagged in ``physicskit.rmt.ensembles.circular``'s module docstring:
COE is NOT "eigenvalues of a Haar-random orthogonal matrix" -- a real
orthogonal matrix's eigenvalues are either real (+-1) or complex-
conjugate pairs on the unit circle, not the generic uniform-phase
structure a circular ensemble requires. This module provides that
other, genuinely different object directly, rather than leaving it as
only a cautionary note: the raw-matrix generators themselves
(``haar_orthogonal``, ``haar_unitary``, ``haar_symplectic``) live in
``physicskit.rmt.utils.haar``, since what they return is the group element
itself (useful directly as a random quantum gate / change of basis for
circuit-simulation and randomized-benchmarking use cases) rather than
an eigenvalue-only product -- the one architectural mismatch with every
other ensemble in this package, whose ``.sample()`` is built around
eigenvalues as the primary product (see ``physicskit.rmt.ensembles.base``).
``HaarUnitary``'s and ``HaarSymplectic``'s eigenvalue-phase statistics
are, respectively, already exactly what CUE and CSE model (see
``physicskit.rmt.ensembles.circular``), so no separate ensemble classes are
added here for those -- only the genuinely new, non-duplicate
Haar-orthogonal eigenvalue ensemble below.

Verified during development (see ``tests/test_haar.py``): unlike CUE
(zero real eigenvalues, generically) and COE (also generically no real
eigenvalues -- see ``physicskit.rmt.ensembles.circular``), a raw Haar-O(n)
matrix has a genuinely nonzero real-eigenvalue count whose expectation
stays O(1) (order unity, empirically close to 1) rather than vanishing
or growing, as n increases -- structurally distinct from both the
circular-ensemble and the GinOE real-eigenvalue-count stories (the
latter grows like sqrt(2n/pi), Edelman-Kostlan-Shub 1994).
"""

import numpy as np

from ..utils.haar import haar_orthogonal
from .base import MatrixEnsemble


class HaarOrthogonalEnsemble(MatrixEnsemble):
    """Eigenvalues of a raw n x n Haar-random orthogonal matrix (O(n)).

    ``Spectrum.eigenvalues`` holds n complex values per sample (a mix of
    real +-1 eigenvalues and complex-conjugate pairs on the unit
    circle) -- see module docstring for why this is a distinct object
    from :class:`~physicskit.rmt.ensembles.circular.COE`.
    """

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        o = haar_orthogonal(self.n, rng)
        return np.linalg.eigvals(o)

    def natural_scale(self) -> float:
        return 1.0
