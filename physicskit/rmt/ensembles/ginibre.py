"""Ginibre ensembles -- GinOE (real), GinUE (complex), GinSE (quaternion).

References
----------
J. Ginibre, "Statistical Ensembles of Complex, Quaternion, and Real
Matrices", J. Math. Phys. 6 (1965) 440.
A. Edelman, E. Kostlan, S. Shub, "How many eigenvalues of a random
matrix are real?", J. Amer. Math. Soc. 7 (1994) 247 -- the real-Ginibre
real-eigenvalue-count result, validated (against its large-n asymptotic
form, not the exact finite-n closed form -- see
``physicskit.rmt.stats.real_ginibre`` for why) by
``physicskit.rmt.validation.RealGinibreEigenvalueCount``.

Unlike the Hermitian/unitary ensembles built so far, Ginibre matrices
have NO symmetry constraint at all -- just i.i.d. entries -- so their
eigenvalues are genuinely complex and spread over a 2-D region (the unit
disk, asymptotically), not confined to the real line or unit circle.
``Spectrum.eigenvalues`` for these ensembles is therefore a complex
array; ``natural_scale`` (sqrt(n)) rescales eigenvalues onto the unit
disk (Ginibre's exact result, not merely asymptotic, for finite n up to
this fixed rescaling).

Real vs. complex vs. quaternion Ginibre are NOT interchangeable dtypes
of the same construction -- they have qualitatively different eigenvalue
structure::

    GinUE (complex): eigenvalues spread generically over the disk.
    GinOE (real): a mix of purely real eigenvalues and complex-conjugate
        pairs (the number of real eigenvalues grows like sqrt(2n/pi),
        Edelman-Kostlan-Shub 1994) -- a vanishing fraction of n as
        n -> infinity, but a structurally distinct, celebrated result
        with no Hermitian-ensemble analogue.
    GinSE (quaternion): a real quaternion matrix, embedded as a 2n x 2n
        complex matrix via the standard quaternion-to-2x2-complex-block
        map, has exactly 2n eigenvalues occurring in complex-conjugate
        pairs (z, z-bar) -- but, importantly, this is NOT the same kind
        of degeneracy as GSE/CSE (where the paired eigenvalues are
        *exactly identical*, a true Kramers symmetry): here z and
        z-bar are genuinely distinct eigenvalues whenever z is not real,
        so GinSE keeps all 2n eigenvalues rather than de-duplicating.
"""

import numpy as np

from .base import MatrixEnsemble


class GinOE(MatrixEnsemble):
    """Real Ginibre ensemble.

    ``Spectrum.eigenvalues`` holds n complex values per sample (purely
    real entries have zero imaginary part).
    """

    beta: float = 1

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        x = rng.standard_normal((self.n, self.n))
        return np.linalg.eigvals(x).astype(complex)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


class GinUE(MatrixEnsemble):
    """Complex Ginibre ensemble."""

    beta: float = 2

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        x = (rng.standard_normal((self.n, self.n)) + 1j * rng.standard_normal((self.n, self.n))) / np.sqrt(2.0)
        return np.linalg.eigvals(x)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


class GinSE(MatrixEnsemble):
    """Quaternion Ginibre ensemble.

    ``n`` is the quaternion matrix dimension; each sample produces 2*n
    complex eigenvalues (genuinely distinct conjugate pairs -- see
    module docstring for why these are NOT de-duplicated, unlike GSE/CSE).
    """

    beta: float = 4

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        a = rng.standard_normal((n, n))
        b = rng.standard_normal((n, n))
        c = rng.standard_normal((n, n))
        d = rng.standard_normal((n, n))
        # Vectorized quaternion-to-2x2-complex-block embedding (verified
        # against the direct per-block loop construction to give an
        # identical matrix): strided assignment places each block
        # [[a+bi, c+di], [-c+di, a-bi]] / 2 directly, without a Python
        # loop over n^2 blocks. The /2 normalizes entry scale to match
        # the same unit-disk circular law radius as GinOE/GinUE after
        # the shared sqrt(n) natural_scale -- verified numerically (see
        # design notes / test suite).
        m = np.zeros((2 * n, 2 * n), dtype=complex)
        m[0::2, 0::2] = (a + 1j * b) / 2.0
        m[0::2, 1::2] = (c + 1j * d) / 2.0
        m[1::2, 0::2] = (-c + 1j * d) / 2.0
        m[1::2, 1::2] = (a - 1j * b) / 2.0
        return np.linalg.eigvals(m)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)
