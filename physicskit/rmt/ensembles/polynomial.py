"""Polynomial ensembles -- squared singular values of a product of
independent Ginibre matrices; a genuine BIORTHOGONAL ensemble (Borodin's
general determinantal framework) whenever more than one factor is used.

References
----------
A. B. J. Kuijlaars, D. Stivigny, "Singular values of products of random
matrices and polynomial ensembles", Random Matrices Theory Appl. 3
(2014) 1450011 -- the "polynomial ensemble" terminology, and the theorem
that products of Ginibre matrices realize one, with explicit
Meijer-G-function biorthogonal weight functions.
A. Borodin, "Biorthogonal ensembles", Nucl. Phys. B 536 (1998) 704 --
the general determinantal framework: joint density
prod_{i<j}(x_i-x_j) * det[phi_k(x_j)] * det[psi_k(x_j)], of which the
classical orthogonal-polynomial ensembles (a single factor, L=1 below)
are the special case phi_k = psi_k (true orthogonal polynomials).
E. Strahov, "Differential equations for singular values of products of
Ginibre random matrices", J. Phys. A 47 (2014) 325203 -- the explicit
non-orthogonal-polynomial biorthogonal kernel for L >= 2, confirming
that construction is NOT reducible to a classical orthogonal-polynomial
ensemble.
G. Akemann, J. R. Ipsen, M. Kieburg, "Products of rectangular random
matrices: singular values and progressive scattering", Phys. Rev. E 88
(2013) 052118.
K. A. Penson, K. Zyczkowski, "Product of Ginibre matrices: Fuss-Catalan
and Raney distributions", Phys. Rev. E 83 (2011) 061118 -- the
Fuss-Catalan limiting law and moment formula used for validation here.

Construction
------------
X = X_1 @ X_2 @ ... @ X_L, each X_k an independent n x n Ginibre matrix
(real, beta=1, or complex, beta=2). This ensemble's eigenvalues are the
squared singular values of X, i.e. the eigenvalues of X^dagger @ X.

L=1 recovers the classical Wishart/Laguerre ensemble EXACTLY (a genuine
orthogonal-polynomial ensemble, phi_k=psi_k in Borodin's framework) --
used as a strong validation anchor in the test suite (a direct match
against the already-validated ``LaguerreBetaEnsemble``/
``MarchenkoPastur`` machinery), not a separate approximation.

L >= 2 is a GENUINE biorthogonal ensemble: Kuijlaars-Stivigny prove the
joint eigenvalue density still has the determinantal
Delta(x) * det[phi_k(x_j)] * det[psi_k(x_j)] form (making it a
"polynomial ensemble"), but Strahov shows the correlation kernel is
provably NOT the reproducing kernel of any single family of orthogonal
polynomials once L >= 2 -- phi_k and psi_k are then genuinely different
function families (built from Meijer G-functions), the defining feature
that distinguishes a biorthogonal ensemble from the classical
orthogonal-polynomial ensembles making up most of the rest of this
package. ``num_factors`` therefore defaults to 2 here, not 1, so the
default instance actually exhibits this genuine biorthogonal structure
rather than silently reducing to the classical case.

After the standard n**L rescaling, the limiting eigenvalue distribution
is the Fuss-Catalan law with parameter L (Penson-Zyczkowski 2011):
support [0, (L+1)**(L+1) / L**L] and exact moments

    m_k = binom((L+1)*k, k) / (L*k + 1)

(L=1 recovers the Marchenko-Pastur moments at gamma=1, i.e. the Catalan
numbers, as it must). ``natural_scale`` divides by this n**L factor
accordingly -- both the scaling exponent and the exact moment formula
were verified numerically during development, at L=1..4, against Monte
Carlo before being used here (not assumed from memory) -- see
``tests/test_polynomial.py``.

Only beta in {1, 2} are supported here (real or complex Ginibre
factors), matching the precedent set by
``physicskit.rmt.ensembles.universality.GeneralWignerEnsemble``: a quaternionic
(beta=4) product-of-Ginibre construction is a further, separate
extension not implemented here.
"""

from math import comb

import numpy as np

from .base import MatrixEnsemble


def fuss_catalan_moment(k: int, num_factors: int) -> float:
    """Exact k-th moment of the Fuss-Catalan(L) distribution -- the
    limiting law of a product of L independent square Ginibre matrices'
    squared singular values, after the n**L rescaling.

    m_k = binom((L+1)*k, k) / (L*k + 1). L=1 recovers the standard
    Marchenko-Pastur moments at gamma=1 (Catalan numbers).
    """
    return comb((num_factors + 1) * k, k) / (num_factors * k + 1)


class PolynomialEnsemble(MatrixEnsemble):
    """Squared singular values of a product of ``num_factors`` independent
    n x n Ginibre matrices -- a polynomial/biorthogonal ensemble (see
    module docstring).

    Parameters
    ----------
    n : int
    num_factors : int, optional
        Number of independent Ginibre factors L (default 2, the
        smallest case exhibiting genuine biorthogonal, non-orthogonal-
        polynomial structure; L=1 recovers the classical Wishart/
        Laguerre ensemble exactly).
    beta : int, optional
        1 (real Ginibre factors) or 2 (complex Ginibre factors).
    """

    def __init__(
        self,
        n: int,
        num_factors: int = 2,
        beta: int = 2,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if num_factors < 1:
            raise ValueError(f"num_factors must be >= 1, got {num_factors}")
        if beta not in (1, 2):
            raise ValueError(f"beta must be 1 or 2, got {beta}")
        super().__init__(n, seed=seed)
        self.num_factors = num_factors
        self.beta: float = beta

    def _sample_factor(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        if self.beta == 1:
            return rng.standard_normal((n, n))
        return (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        x = self._sample_factor(rng)
        for _ in range(self.num_factors - 1):
            x = x @ self._sample_factor(rng)
        return np.linalg.eigvalsh(x.conj().T @ x)

    def natural_scale(self) -> float:
        return float(self.n) ** self.num_factors
