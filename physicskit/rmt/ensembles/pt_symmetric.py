"""PT-symmetric (pseudo-Hermitian) random matrix ensembles.

References
----------
C. M. Bender, S. Boettcher, "Real Spectra in Non-Hermitian Hamiltonians
Having PT Symmetry", Phys. Rev. Lett. 80 (1998) 5243 -- the original
PT-symmetric-quantum-mechanics observation this construction realizes
as a random matrix model.
A. Mostafazadeh, "Pseudo-Hermiticity versus PT-symmetry", J. Math. Phys.
43 (2002) 205, 2814, 3944 -- the equivalence (for a diagonalizable H
with real spectrum) between PT-symmetry and pseudo-Hermiticity
P H P^{-1} = H^dagger for some Hermitian, invertible P, which is the
concrete condition realized here (P chosen as a fixed, real, diagonal
+-1 "signature" matrix, so P = P^dagger = P^{-1}).

Construction: partition the n = p + q dimensions into a "+1" block
(size p) and a "-1" block (size q), P = diag(I_p, -I_q). Writing H in
the matching 2x2 block form,

    H = [[A, B], [C, D]]

the pseudo-Hermiticity condition P H P = H^dagger forces A = A^dagger
(p x p Hermitian), D = D^dagger (q x q Hermitian), and C = -B^dagger
(NOT the +B^dagger a genuinely Hermitian H would need) -- verified
numerically here (``max|PHP - H^dagger|`` to machine precision) before
being trusted. A, D are drawn as independent GOE/GUE-type Hermitian
blocks; B is an independent p x q Ginibre-type block scaled by a
non-Hermiticity/coupling parameter ``g``.

Physics (verified numerically during development, see
``tests/test_pt_symmetric.py``): at g=0, H is block-diagonal (A
directly summed with D), so its spectrum is trivially real (each block
is Hermitian) -- the fully "unbroken" PT-symmetric phase. Since H is
similar to H^dagger (via the invertible P), its eigenvalue set is
always closed under complex conjugation: every eigenvalue is either
real or part of an exact complex-conjugate pair -- never an isolated
complex value. As g increases from 0, an increasing fraction of
eigenvalues collide in real-conjugate pairs and move off the real axis
as complex-conjugate pairs (each such collision is an "exceptional
point": both the eigenvalues AND their eigenvectors coalesce there,
unlike an ordinary level crossing) -- the celebrated PT-symmetry-
breaking transition. See ``physicskit.rmt.stats.pt_symmetric`` for the fraction-
of-real-eigenvalues statistic that tracks this transition.
"""

from __future__ import annotations

import numpy as np

from .base import MatrixEnsemble


def _dense_hermitian_block(n: int, rng: np.random.Generator, complex_entries: bool) -> np.ndarray:
    if complex_entries:
        x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
        return (x + x.conj().T) / 2.0
    x = rng.standard_normal((n, n))
    return (x + x.T) / np.sqrt(2.0)


class PTSymmetricEnsemble(MatrixEnsemble):
    """Pseudo-Hermitian (PT-symmetric-realizing) random matrix ensemble
    (see module docstring). ``n`` (inherited from ``MatrixEnsemble``) is
    the total dimension p + q.

    Parameters
    ----------
    p : int
        Size of the "+1"-signature block.
    q : int
        Size of the "-1"-signature block.
    g : float
        Non-Hermiticity / coupling strength scaling the off-diagonal
        block B (g=0: block-diagonal, trivially real spectrum; larger
        g: more eigenvalues driven into complex-conjugate pairs).
    beta : int, optional
        1 (real blocks) or 2 (complex Hermitian blocks, default).
    seed : int, numpy.random.Generator, or None, optional
    """

    def __init__(
        self,
        p: int,
        q: int,
        g: float,
        beta: int = 2,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if beta not in (1, 2):
            raise ValueError(f"PTSymmetricEnsemble only supports beta in (1, 2), got {beta}")
        if g < 0:
            raise ValueError(f"g (coupling strength) must be >= 0, got {g}")
        super().__init__(p + q, seed=seed)
        self.p = p
        self.q = q
        self.g = float(g)
        self.beta: float = beta

    def _sample_matrix(self, rng: np.random.Generator) -> np.ndarray:
        complex_entries = self.beta == 2
        a = _dense_hermitian_block(self.p, rng, complex_entries)
        d = _dense_hermitian_block(self.q, rng, complex_entries)
        if complex_entries:
            b = self.g * (rng.standard_normal((self.p, self.q)) + 1j * rng.standard_normal((self.p, self.q))) / np.sqrt(2.0)
        else:
            b = self.g * rng.standard_normal((self.p, self.q))
        c = -b.conj().T
        top = np.hstack([a, b])
        bottom = np.hstack([c, d])
        return np.vstack([top, bottom])

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return np.linalg.eigvals(self._sample_matrix(rng)).astype(complex)

    def natural_scale(self) -> float:
        # No single universal rescaling: unlike the Hermitian ensembles,
        # this ensemble's spectral shape genuinely changes character
        # (real -> partially complex) as g varies, so there is no fixed
        # limiting law to normalize against.
        return 1.0
