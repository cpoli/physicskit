"""Bogoliubov-de Gennes (BdG) / Altland-Zirnbauer superconductor classes:
D, C, CI, DIII.

These extend Dyson's threefold way (classes A, AI, AII) by adding
particle-hole (charge-conjugation) symmetry, appropriate to
non-interacting fermionic Hamiltonians in disordered/chaotic
superconductors. Together with the threefold way and the three chiral
classes (AIII, BDI, CII -- QCD-type, realized elsewhere in this package
via the Wishart/Laguerre singular-value connection), these four complete
Altland and Zirnbauer's "tenfold way".

References
----------
A. Altland, M. R. Zirnbauer, "Nonstandard symmetry classes in
mesoscopic normal-superconducting hybrid structures", Phys. Rev. B 55
(1997) 1142.
M. A. Stephanov, J. J. M. Verbaarschot, T. Wettig, "Random Matrices"
(Wiley Encyclopedia of Electrical and Electronics Engineering, 2005),
arXiv:hep-ph/0509286 -- the block structure and (alpha, beta) table for
classes D and C used here.
M. Stolz, "Fluctuations of Wigner-type random matrices associated with
symmetric spaces of class DIII and CI", arXiv:1707.03793 -- the
explicit, unambiguous matrix-space definitions for DIII and CI used
here (Section 2), stated in purely mathematical terms independent of
any physics convention.

Construction
------------
All four classes have the :math:`2N \\times 2N` Hermitian block structure

.. math::

   H = \\begin{pmatrix} A & B \\\\ B^\\dagger & -A^T \\end{pmatrix}

(class D, C) or, equivalently in the special case where :math:`B`
happens to equal its own conjugate transpose (true for the specific
reality/symmetry combinations used in CI and DIII),

.. math::

   H = \\begin{pmatrix} X_1 & X_2 \\\\ X_2 & -X_1 \\end{pmatrix}

(class CI, DIII), where::

    Class D    (beta=2): A complex Hermitian (N x N), B complex antisymmetric.
    Class C    (beta=2): A complex Hermitian (N x N), B complex symmetric.
    Class CI   (beta=1): X1, X2 real symmetric (N x N) -- H is real symmetric.
    Class DIII (beta=4): X1, X2 purely imaginary skew-symmetric (N x N)
                          -- i.e. i times a real antisymmetric matrix,
                          which is itself Hermitian.

or, in symmetry-constraint form,

.. math::

   \\text{D: } A^\\dagger = A,\\ B^T = -B \\qquad
   \\text{C: } A^\\dagger = A,\\ B^T = B

.. math::

   \\text{CI: } X_1^T = X_1,\\ X_2^T = X_2,\\ X_1, X_2 \\in \\mathbb{R} \\qquad
   \\text{DIII: } X_1^T = -X_1,\\ X_2^T = -X_2,\\ X_1, X_2 \\in i\\mathbb{R}

This block structure has a built-in particle-hole (charge-conjugation)
symmetry

.. math::

   \\mathcal{C} H \\mathcal{C}^{-1} = -H, \\qquad
   \\mathcal{C} = \\tau_x \\mathcal{K}, \\qquad
   \\tau_x = \\begin{pmatrix} 0 & I_N \\\\ I_N & 0 \\end{pmatrix}

(:math:`\\mathcal{K}` complex conjugation) that forces the spectrum to be
symmetric about zero: if :math:`\\lambda` is an eigenvalue, so is
:math:`-\\lambda`. This is NOT a degeneracy (unlike the Kramers doubling
in GSE/CSE) -- :math:`\\lambda` and :math:`-\\lambda` are generically
distinct eigenvalues. Verified during development: all four
constructions are exactly Hermitian (to machine precision) and exhibit
this :math:`\\pm\\lambda` pairing to machine precision -- see
``tests/test_bdg.py``. Going further (validating the exact bulk/edge
eigenvalue density, which follows Laguerre-type statistics in the
variable :math:`\\lambda^2` per the :math:`(\\alpha, \\beta)` table in
Stephanov-Verbaarschot-Wettig) is deferred; only the structural
properties above are validated here.
"""

import numpy as np

from .base import MatrixEnsemble


def _dense_hermitian(n: int, rng: np.random.Generator, complex_entries: bool) -> np.ndarray:
    if complex_entries:
        x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
        return (x + x.conj().T) / 2.0
    x = rng.standard_normal((n, n))
    return (x + x.T) / np.sqrt(2.0)


def _dense_symmetric(n: int, rng: np.random.Generator, complex_entries: bool) -> np.ndarray:
    if complex_entries:
        x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    else:
        x = rng.standard_normal((n, n))
    return (x + x.T) / np.sqrt(2.0)


def _dense_antisymmetric(n: int, rng: np.random.Generator, complex_entries: bool) -> np.ndarray:
    if complex_entries:
        x = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    else:
        x = rng.standard_normal((n, n))
    return (x - x.T) / np.sqrt(2.0)


class BdGClassD(MatrixEnsemble):
    """Altland-Zirnbauer class D (beta=2).

    .. math::

       H = \\begin{pmatrix} A & B \\\\ B^\\dagger & -A^T \\end{pmatrix},
       \\qquad A^\\dagger = A,\\ \\ B^T = -B

    with :math:`A` complex Hermitian and :math:`B` complex antisymmetric
    (both :math:`N \\times N`). ``n`` is the block size :math:`N`; each
    sample is a :math:`2N \\times 2N` Hermitian matrix with :math:`2N`
    eigenvalues occurring in :math:`\\pm\\lambda` pairs."""

    beta: float = 2

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        a = _dense_hermitian(n, rng, complex_entries=True)
        b = _dense_antisymmetric(n, rng, complex_entries=True)
        h = np.block([[a, b], [b.conj().T, -a.T]])
        h = (h + h.conj().T) / 2.0  # symmetrize away numerical asymmetry
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


class BdGClassC(MatrixEnsemble):
    """Altland-Zirnbauer class C (beta=2).

    .. math::

       H = \\begin{pmatrix} A & B \\\\ B^\\dagger & -A^T \\end{pmatrix},
       \\qquad A^\\dagger = A,\\ \\ B^T = B

    with :math:`A` complex Hermitian and :math:`B` complex symmetric
    (both :math:`N \\times N`). ``n`` is the block size :math:`N`; each
    sample is a :math:`2N \\times 2N` Hermitian matrix with :math:`2N`
    eigenvalues occurring in :math:`\\pm\\lambda` pairs."""

    beta: float = 2

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        a = _dense_hermitian(n, rng, complex_entries=True)
        b = _dense_symmetric(n, rng, complex_entries=True)
        h = np.block([[a, b], [b.conj().T, -a.T]])
        h = (h + h.conj().T) / 2.0
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


class BdGClassCI(MatrixEnsemble):
    """Altland-Zirnbauer class CI (beta=1).

    .. math::

       H = \\begin{pmatrix} X_1 & X_2 \\\\ X_2 & -X_1 \\end{pmatrix},
       \\qquad X_1^T = X_1,\\ \\ X_2^T = X_2,\\ \\ X_1, X_2 \\in \\mathbb{R}^{N \\times N}

    so :math:`H` is real symmetric (:math:`2N \\times 2N`)."""

    beta: float = 1

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        x1 = _dense_symmetric(n, rng, complex_entries=False)
        x2 = _dense_symmetric(n, rng, complex_entries=False)
        h = np.block([[x1, x2], [x2, -x1]])
        h = (h + h.T) / 2.0
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


class BdGClassDIII(MatrixEnsemble):
    """Altland-Zirnbauer class DIII (beta=4).

    .. math::

       H = \\begin{pmatrix} X_1 & X_2 \\\\ X_2 & -X_1 \\end{pmatrix},
       \\qquad X_1 = iY_1,\\ X_2 = iY_2,\\ \\ Y_1^T = -Y_1,\\ Y_2^T = -Y_2,\\ \\ Y_1, Y_2 \\in \\mathbb{R}^{N \\times N}

    i.e. :math:`X_1, X_2` are purely imaginary skew-symmetric
    (:math:`N \\times N`, hence themselves Hermitian). :math:`H` is
    complex Hermitian (:math:`2N \\times 2N`)."""

    beta: float = 4

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        y1 = _dense_antisymmetric(n, rng, complex_entries=False)
        y2 = _dense_antisymmetric(n, rng, complex_entries=False)
        x1, x2 = 1j * y1, 1j * y2
        h = np.block([[x1, x2], [x2, -x1]])
        h = (h + h.conj().T) / 2.0
        return np.linalg.eigvalsh(h)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)
