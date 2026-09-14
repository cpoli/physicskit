"""Girko ensembles -- universality generalizations of the Ginibre
ensembles.

References
----------
V. L. Girko, "Circular law", Theory Probab. Appl. 29 (1984) 694 -- the
circular law holds for ANY i.i.d. entry distribution (mean 0, unit
variance), not just Gaussian.
Z. D. Bai, "Circular law", Ann. Probab. 25 (1997) 494 -- fixed gaps in
Girko's original proof.
T. Tao, V. Vu (with an appendix by M. Krishnapur), "Random matrices:
Universality of ESDs and the circular law", Ann. Probab. 38 (2010) 2023
-- relaxed the moment condition to finite variance.
V. L. Girko, "The elliptic law", Theory Probab. Appl. 30 (1985) 677 --
matrices whose (X_ij, X_ji) entry pairs are correlated have eigenvalues
filling an ellipse rather than a disk.
A. Naumov, "Elliptic law for random matrices", arXiv:1201.1639 (2012) --
rigorous elliptic law for general (non-Gaussian) entry distributions.
Y. V. Fyodorov, B. A. Khoruzhenko, H.-J. Sommers, "Almost-Hermitian
random matrices: eigenvalue density in the complex plane", Phys. Lett.
A 226 (1997) 46 -- the Gaussian special case ("elliptic Ginibre"), which
:class:`GirkoElliptic` recovers for its default entry distribution --
per the design notes, FKS is a parametrization of the general Girko
result below, not a separate construction.

Both ensembles reuse ``physicskit.rmt.ensembles.universality``'s entry-sampler
convention (``uniform_unit_variance``, ``rademacher``,
``exponential_centered_unit_variance``) -- the same non-Gaussian entry
distributions already used there to demonstrate Wigner (Hermitian)
universality now demonstrate its non-Hermitian analogue: the circular
and elliptic laws depend only on the first two moments of the entry
distribution, not on Gaussianity.

Unlike every Dyson-indexed ensemble elsewhere in this package,
:class:`IIDEnsemble` and :class:`GirkoElliptic` have NO beta:
Girko universality is a different, orthogonal axis from Dyson's
threefold way -- it says the *macroscopic* eigenvalue law
(circular/elliptic) is unchanged across entry distributions, not that
the *exact* joint eigenvalue density matches a beta-ensemble formula
(only Gaussian entries have that property). ``Spectrum.beta`` is
therefore left ``None`` for both, inherited from the base class.
"""

from collections.abc import Callable

import numpy as np

from .base import MatrixEnsemble

EntrySampler = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]


def standard_normal(rng: np.random.Generator, size: tuple[int, ...]) -> np.ndarray:
    """i.i.d. standard Gaussian entries -- mean 0, variance 1. Default
    entry distribution for :class:`GirkoElliptic` (the FKS elliptic
    Ginibre special case); also usable directly with :class:`IIDEnsemble`
    to recover plain Ginibre-law behavior via the general construction."""
    return rng.standard_normal(size)


class IIDEnsemble(MatrixEnsemble):
    """Girko's circular law: eigenvalues of an n x n matrix with i.i.d.
    mean-zero, unit-variance entries (real or complex) converge to the
    uniform distribution on the unit disk after the sqrt(n) rescaling --
    exactly the GinOE/GinUE limiting law, but for ANY entry distribution
    satisfying those two moment conditions, not just Gaussian.

    Parameters
    ----------
    n : int
    entry_sampler : callable
        ``entry_sampler(rng, size)`` returning i.i.d. mean-zero,
        unit-variance real samples of the given shape -- see
        ``standard_normal`` above and
        ``physicskit.rmt.ensembles.universality.uniform_unit_variance``,
        ``rademacher``, ``exponential_centered_unit_variance`` for
        ready-made options.
    complex_entries : bool, optional
        If True (default), each entry is
        ``(entry_sampler + 1j*entry_sampler) / sqrt(2)`` -- matching
        GinUE's construction. If False, entries are real -- matching
        GinOE, including its real-eigenvalue anomaly at any finite n
        (a property of real matrices generally, not just Gaussian ones).
    """

    def __init__(
        self,
        n: int,
        entry_sampler: EntrySampler,
        complex_entries: bool = True,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        super().__init__(n, seed=seed)
        self.entry_sampler = entry_sampler
        self.complex_entries = complex_entries

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        n = self.n
        if self.complex_entries:
            x = (self.entry_sampler(rng, (n, n)) + 1j * self.entry_sampler(rng, (n, n))) / np.sqrt(2.0)
        else:
            x = self.entry_sampler(rng, (n, n))
        return np.linalg.eigvals(x).astype(complex)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)


def _correlated_pair_matrix(n: int, rho: float, entry_sampler: EntrySampler, rng: np.random.Generator) -> np.ndarray:
    """n x n real matrix whose off-diagonal entry pairs (X_ij, X_ji)
    (i != j) each have unit variance and correlation ``rho``.

    Construction: independent A, B ~ entry_sampler; for i < j,
        X_ij = c1*A_ij + c2*B_ij
        X_ji = c1*A_ij - c2*B_ij
    with c1 = sqrt((1+rho)/2), c2 = sqrt((1-rho)/2). Directly verifiable:
    Var(X_ij) = Var(X_ji) = c1^2 + c2^2 = 1 and
    Cov(X_ij, X_ji) = c1^2 - c2^2 = rho, for any entry_sampler with
    mean 0, unit variance -- not just Gaussian (Naumov 2012).

    Diagonal entries are set to X_ii = c1*sqrt(2)*A_ii, giving
    Var(X_ii) = 1+rho -- chosen (rather than a fixed unit variance) so
    the construction degenerates *exactly* at both endpoints: at rho=1
    (c1=1) this is sqrt(2)*A_ii, the standard Wigner/GOE
    variance-2 diagonal convention used elsewhere in this package (see
    ``physicskit.rmt.ensembles.universality.GeneralWignerEnsemble``), and at
    rho=-1 (c1=0) the diagonal is exactly zero -- required for X to be
    exactly antisymmetric there (X_ii = -X_ii forces X_ii = 0; no
    distribution of nonzero variance can satisfy that identically).
    """
    a = entry_sampler(rng, (n, n))
    b = entry_sampler(rng, (n, n))
    c1 = np.sqrt((1.0 + rho) / 2.0)
    c2 = np.sqrt((1.0 - rho) / 2.0)
    upper = np.triu(c1 * a + c2 * b, k=1)
    lower = np.triu(c1 * a - c2 * b, k=1).T
    diagonal = np.diag(c1 * np.sqrt(2.0) * np.diag(a))
    return upper + lower + diagonal


class GirkoElliptic(MatrixEnsemble):
    """Girko's elliptic law: eigenvalues of an n x n real matrix whose
    (X_ij, X_ji) entry pairs are correlated at ``rho`` (each individually
    unit variance) fill an ellipse with semi-axes (1+rho) along the real
    axis and (1-rho) along the imaginary axis, after the sqrt(n)
    rescaling.

    rho=0 recovers plain i.i.d. entries -- Girko's circular law, a disk
    of radius 1, the same limiting law as :class:`IIDEnsemble` with
    ``complex_entries=False``. The endpoints rho=+-1 are exact
    (not just asymptotic) degenerate limits, checkable at any finite n:
    rho=1 makes X exactly symmetric (X_ij = X_ji), so eigenvalues are
    exactly real (recovering the real-entries Wigner/semicircle case,
    the ellipse collapsed onto the real segment [-2, 2]); rho=-1 makes X
    exactly antisymmetric, so eigenvalues are exactly purely imaginary
    (real antisymmetric matrices have eigenvalues 0 or imaginary
    conjugate pairs -- the ellipse collapsed onto the imaginary axis).

    Parameters
    ----------
    n : int
    rho : float
        Entry-pair correlation, in [-1, 1].
    entry_sampler : callable, optional
        ``entry_sampler(rng, size)`` returning i.i.d. mean-zero,
        unit-variance real samples (default: ``standard_normal``, i.e.
        the Fyodorov-Khoruzhenko-Sommers elliptic Ginibre special case).
    """

    def __init__(
        self,
        n: int,
        rho: float,
        entry_sampler: EntrySampler | None = None,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if not -1.0 <= rho <= 1.0:
            raise ValueError(f"rho must be in [-1, 1], got {rho}")
        super().__init__(n, seed=seed)
        self.rho = float(rho)
        self.entry_sampler = entry_sampler or standard_normal

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        x = _correlated_pair_matrix(self.n, self.rho, self.entry_sampler, rng)
        return np.linalg.eigvals(x).astype(complex)

    def natural_scale(self) -> float:
        return np.sqrt(self.n)
