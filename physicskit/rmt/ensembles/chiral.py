"""Chiral ensembles -- chGOE, chGUE, chGSE (Altland-Zirnbauer classes
BDI, AIII, CII): block off-diagonal Hamiltonians with an anticommuting
chiral symmetry, the QCD-type ("Dirac operator") random matrix ensembles
of Verbaarschot and Zahed. Together with the threefold way (A, AI, AII,
realized in ``physicskit.rmt.ensembles.gaussian``) and the four BdG classes
(``physicskit.rmt.ensembles.bdg``), these complete Altland-Zirnbauer's "tenfold
way".

References
----------
J. J. M. Verbaarschot, I. Zahed, "Spectral density of the QCD Dirac
operator near zero virtuality", Phys. Rev. Lett. 70 (1993) 3852.
J. J. M. Verbaarschot, "The spectrum of the QCD Dirac operator and
chiral random matrix theory", Phys. Rev. Lett. 72 (1994) 2531.
A. Altland, M. R. Zirnbauer, "Nonstandard symmetry classes in
mesoscopic normal-superconducting hybrid structures", Phys. Rev. B 55
(1997) 1142 -- classes BDI, AIII, CII within the tenfold way.
M. A. Stephanov, J. J. M. Verbaarschot, T. Wettig, "Random Matrices",
arXiv:hep-ph/0509286 -- the chiral <-> Wishart/Laguerre equivalence
used here.

Construction
------------
A chiral Hamiltonian has the block off-diagonal form

.. math::

   H = \\begin{pmatrix} 0 & W \\\\ W^\\dagger & 0 \\end{pmatrix}

where :math:`W` is a rectangular :math:`N \\times (N+\\nu)` matrix
(:math:`\\nu \\geq 0` the topological index -- the number of exact zero
modes, i.e. the QCD Dirac operator's index), drawn with i.i.d. entries
from the Dyson symmetry class matching beta: real (beta=1, class BDI,
chGOE), complex (beta=2, class AIII, chGUE), or quaternionic (beta=4,
class CII, chGSE). :math:`H` is Hermitian by construction and
anticommutes with the chiral "gamma_5" operator
:math:`\\Gamma = \\mathrm{diag}(I_N, -I_{N+\\nu})`:

.. math::

   \\Gamma H \\Gamma = -H

-- the defining symmetry of this class (verified directly, to machine
precision, in ``tests/test_chiral.py``), forcing the spectrum to be
exactly symmetric about zero -- like BdG's particle-hole symmetry
(``physicskit.rmt.ensembles.bdg``), but via an ANTIcommuting rather than a
commuting operator, and via a block off-diagonal rather than a block
diagonal-plus-symmetric structure.

Because

.. math::

   H^2 = \\begin{pmatrix} WW^\\dagger & 0 \\\\ 0 & W^\\dagger W \\end{pmatrix},

:math:`H`'s nonzero eigenvalues are exactly :math:`\\pm\\sigma_k`, the
singular values of :math:`W` -- and :math:`W^\\dagger W` is EXACTLY a
beta-Wishart/Laguerre matrix (``physicskit.rmt.ensembles.wishart``).
This is why the chiral ensembles need no new sampling machinery: unlike
BdG, which needed genuinely new dense constructions, chiral eigenvalues
are generated here by reusing
``physicskit.rmt.utils.tridiagonal.sample_laguerre_beta_eigenvalues`` directly
and taking its square root -- exact in distribution (via the same
Dumitriu-Edelman generalized-beta mechanism used by
``HermiteBetaEnsemble``/``LaguerreBetaEnsemble``), without ever forming
or diagonalizing a dense matrix. This also means that, exactly as
``HermiteBetaEnsemble``'s GSE (beta=4) returns n eigenvalues (not the 2n
a dense quaternion embedding would give -- see
``physicskit.rmt.ensembles.ginibre``'s GinSE for that contrast), the chiral
ensembles here return ``n`` +-sigma pairs at every beta, including 4 --
NOT a doubled count from an explicit quaternion embedding.

The ``nu`` exact zero eigenvalues (the topological zero modes) are
appended directly, since W^dagger @ W (an (N+nu) x (N+nu) matrix of rank
<= N) has exactly ``nu`` zero eigenvalues whenever nu > 0 -- not merely
small, exactly zero, reflecting the index-theorem structure underlying
this construction (the same reason a private ``_dense_chiral_hamiltonian``
helper below, used only for direct structural verification in the test
suite, checks this on the literal block matrix rather than trusting it
transitively from the Laguerre connection).

Since ``sample_laguerre_beta_eigenvalues`` already returns
Marchenko-Pastur-normalized eigenvalues (support
[(1-sqrt(gamma))^2, (1+sqrt(gamma))^2], gamma = n/(n+nu)),
``natural_scale`` is 1.0 here too (matching
``LaguerreBetaEnsemble.natural_scale``): the nonzero +-sigma_k already
lie in [-(1+sqrt(gamma)), -(1-sqrt(gamma))] union
[(1-sqrt(gamma)), (1+sqrt(gamma))].

Two eigenvalue-convention presentations
----------------------------------------
``ChiralBetaEnsemble``/``chGOE``/``chGUE``/``chGSE`` present the HERMITIAN
gamma_5-Hamiltonian :math:`H` (real eigenvalues :math:`\\pm\\sigma_k` and
:math:`\\nu` exact zeros). The lattice-QCD and chiral-random-matrix-theory
literature, however, usually works directly with the (anti-Hermitian)
EUCLIDEAN DIRAC OPERATOR :math:`D`, related to :math:`H` by

.. math::

   D = iH,

whose eigenvalues are therefore purely imaginary: :math:`\\pm i\\sigma_k`
(the nonzero "virtualities") plus :math:`\\nu` exact zero modes.
``QCDDiracEnsemble``/``QCDDiracGOE``/``QCDDiracGUE``/
``QCDDiracGSE`` below provide exactly this presentation -- a thin
wrapper multiplying ``ChiralBetaEnsemble``'s eigenvalues by ``i``, not a
mathematically distinct construction, provided purely so the eigenvalue
convention matches what papers in that literature actually plot
(Im(lambda_n), the Dirac spectral density rho(lambda)) without the
caller having to remember to do the rotation themselves.
"""

import numpy as np

from ..utils.tridiagonal import sample_laguerre_beta_eigenvalues
from .base import MatrixEnsemble


class ChiralBetaEnsemble(MatrixEnsemble):
    """General beta-chiral ensemble (block off-diagonal, QCD-type Dirac
    operator random matrix).

    .. math::

       H = \\begin{pmatrix} 0 & W \\\\ W^\\dagger & 0 \\end{pmatrix},
       \\qquad \\Gamma H \\Gamma = -H,
       \\qquad \\Gamma = \\mathrm{diag}(I_N, -I_{N+\\nu})

    beta=1, 2, 4 recover chGOE, chGUE, chGSE (Altland-Zirnbauer classes
    BDI, AIII, CII) exactly; any beta > 0 is a valid continuum
    generalization (inherited from the underlying beta-Laguerre
    construction).

    Parameters
    ----------
    n : int
        Number of nonzero +-sigma eigenvalue pairs (the smaller block
        dimension N).
    beta : float
        Dyson index.
    nu : int, optional
        Topological index: number of exact zero eigenvalues (default
        0). The larger block dimension is N + nu.
    """

    def __init__(
        self,
        n: int,
        beta: float,
        nu: int = 0,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if beta <= 0:
            raise ValueError(f"beta must be positive, got {beta}")
        if nu < 0:
            raise ValueError(f"nu (topological index) must be >= 0, got {nu}")
        super().__init__(n, seed=seed)
        self.beta: float = float(beta)
        self.nu = nu

    @property
    def gamma(self) -> float:
        """Aspect ratio :math:`\\gamma = n/(n+\\nu)` controlling the
        Marchenko-Pastur-derived support :math:`[(1-\\sqrt{\\gamma})^2,
        (1+\\sqrt{\\gamma})^2]` of the nonzero :math:`\\pm\\sigma` eigenvalues."""
        return self.n / (self.n + self.nu)

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        m = self.n + self.nu
        squared_singular_values = sample_laguerre_beta_eigenvalues(m, self.n, self.beta, rng)
        sigma = np.sqrt(squared_singular_values)
        zero_modes = np.zeros(self.nu)
        return np.concatenate([-sigma, zero_modes, sigma])

    def natural_scale(self) -> float:
        return 1.0


class chGOE(ChiralBetaEnsemble):
    """Chiral Orthogonal Ensemble (beta=1, Altland-Zirnbauer class BDI).

    .. math::

       H = \\begin{pmatrix} 0 & W \\\\ W^T & 0 \\end{pmatrix}, \\qquad W \\in \\mathbb{R}^{N \\times (N+\\nu)}

    real block off-diagonal Hamiltonian (:math:`W` real)."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=1, nu=nu, seed=seed)


class chGUE(ChiralBetaEnsemble):
    """Chiral Unitary Ensemble (beta=2, Altland-Zirnbauer class AIII).

    .. math::

       H = \\begin{pmatrix} 0 & W \\\\ W^\\dagger & 0 \\end{pmatrix}, \\qquad W \\in \\mathbb{C}^{N \\times (N+\\nu)}

    complex block off-diagonal Hamiltonian (:math:`W` complex) -- the
    QCD Dirac operator random matrix ensemble of Verbaarschot-Zahed."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=2, nu=nu, seed=seed)


class chGSE(ChiralBetaEnsemble):
    """Chiral Symplectic Ensemble (beta=4, Altland-Zirnbauer class CII).

    .. math::

       H = \\begin{pmatrix} 0 & W \\\\ W^\\dagger & 0 \\end{pmatrix}, \\qquad W \\in \\mathbb{H}^{N \\times (N+\\nu)}

    quaternionic block off-diagonal Hamiltonian (:math:`W` quaternionic)."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=4, nu=nu, seed=seed)


class QCDDiracEnsemble(ChiralBetaEnsemble):
    """The QCD Dirac operator

    .. math::

       D = iH

    in the anti-Hermitian convention standard in the lattice-QCD and
    chiral-random-matrix literature (Verbaarschot-Zahed): eigenvalues
    are purely imaginary, :math:`\\pm i\\sigma_k` plus :math:`\\nu` exact
    zero modes, rather than ``ChiralBetaEnsemble``'s real
    :math:`\\pm\\sigma_k` -- see the module docstring section "Two
    eigenvalue-convention presentations". Parameters are identical to
    ``ChiralBetaEnsemble``.
    """

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return 1j * super()._sample_eigenvalues(rng)


class QCDDiracGOE(QCDDiracEnsemble):
    """QCD Dirac operator built on chGOE (beta=1, class BDI): real
    coupling block W, purely imaginary Dirac eigenvalues."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=1, nu=nu, seed=seed)


class QCDDiracGUE(QCDDiracEnsemble):
    """QCD Dirac operator built on chGUE (beta=2, class AIII): complex
    coupling block W, purely imaginary Dirac eigenvalues -- the
    Verbaarschot-Zahed QCD Dirac operator ensemble in its standard
    anti-Hermitian presentation."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=2, nu=nu, seed=seed)


class QCDDiracGSE(QCDDiracEnsemble):
    """QCD Dirac operator built on chGSE (beta=4, class CII):
    quaternionic coupling block W, purely imaginary Dirac eigenvalues."""

    def __init__(self, n: int, nu: int = 0, seed: int | np.random.Generator | None = None) -> None:
        super().__init__(n, beta=4, nu=nu, seed=seed)


def _sample_block(n: int, m: int, rng: np.random.Generator, beta: int) -> np.ndarray:
    """Rectangular N x M coupling block W matching the Dyson symmetry
    class beta, via an EXPLICIT dense (quaternion-embedded, for beta=4)
    construction. Used only for direct structural verification of the
    chiral-symmetry construction in the test suite -- the efficient
    production sampler above (``ChiralBetaEnsemble._sample_eigenvalues``)
    never forms this matrix; see the module docstring for why."""
    if beta == 1:
        return rng.standard_normal((n, m))
    if beta == 2:
        return (rng.standard_normal((n, m)) + 1j * rng.standard_normal((n, m))) / np.sqrt(2.0)
    if beta == 4:
        a = rng.standard_normal((n, m))
        b = rng.standard_normal((n, m))
        c = rng.standard_normal((n, m))
        d = rng.standard_normal((n, m))
        w = np.zeros((2 * n, 2 * m), dtype=complex)
        w[0::2, 0::2] = a + 1j * b
        w[0::2, 1::2] = c + 1j * d
        w[1::2, 0::2] = -c + 1j * d
        w[1::2, 1::2] = a - 1j * b
        return w / np.sqrt(2.0)
    raise ValueError(f"beta must be 1, 2, or 4, got {beta}")


def _dense_chiral_hamiltonian(n: int, nu: int, rng: np.random.Generator, beta: int) -> np.ndarray:
    """Dense block off-diagonal chiral Hamiltonian H = [[0, W], [W^dagger, 0]]
    -- used only for direct structural verification (Hermiticity, exact
    chiral-symmetry anticommutation with Gamma = diag(I, -I), and the
    +-singular-value / exact-zero-mode eigenvalue structure) in the test
    suite. Note beta=4 uses an explicit quaternion embedding here, so its
    block sizes are doubled relative to the (n, n+nu) the fast sampler
    above works with directly -- an independent linear-algebra fact
    about the chiral block structure, not tied to which of the two
    (embedding vs. generalized-beta) conventions is used to realize it.
    """
    m = n + nu
    w = _sample_block(n, m, rng, beta)
    n_dim, m_dim = w.shape
    top = np.hstack([np.zeros((n_dim, n_dim), dtype=complex), w])
    bottom = np.hstack([w.conj().T, np.zeros((m_dim, m_dim), dtype=complex)])
    return np.vstack([top, bottom])
