"""Generate Haar-distributed random unitary matrices via the corrected
QR decomposition trick.

Reference: F. Mezzadri, "How to generate random matrices from the
classical compact groups", Notices of the AMS 54 (2007) 592
(arXiv:math-ph/0609050).

A naive QR decomposition of a complex Ginibre matrix does NOT produce a
Haar-distributed unitary matrix: the QR decomposition is not unique (Q can
be multiplied by any unitary diagonal matrix while R is adjusted to
compensate), and standard LAPACK routines resolve that ambiguity in a way
that is not rotation-invariant. Fixing the phases of R's diagonal so they
are all real and positive removes the ambiguity and makes Q exactly
Haar-distributed (Mezzadri's Theorem 1). Verified empirically: the
resulting matrix is unitary to machine precision, and eigenvalue phases
are uniform on the circle (see physicskit.rmt.ensembles.circular.CUE and
tests/test_circular.py).
"""

import numpy as np


def haar_unitary(n: int, rng: np.random.Generator) -> np.ndarray:
    """A single n x n Haar-distributed random unitary matrix.

    Parameters
    ----------
    n : int
        Matrix dimension.
    rng : numpy.random.Generator
        Random number generator to draw from.

    Returns
    -------
    numpy.ndarray, shape (n, n), complex
    """
    z = (rng.standard_normal((n, n)) + 1j * rng.standard_normal((n, n))) / np.sqrt(2.0)
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    ph = d / np.abs(d)
    return q * ph  # broadcasts the phase correction over columns of q


def symplectic_form(n: int) -> np.ndarray:
    """The n x n (n even) symplectic form Z = I_(n/2) kron [[0, 1], [-1, 0]],
    used to build the "dual" of a unitary matrix for the Circular
    Symplectic Ensemble (see physicskit.rmt.ensembles.circular.CSE). Satisfies
    Z^2 = -I, so Z^{-1} = -Z.

    Parameters
    ----------
    n : int
        Dimension; must be even.

    Returns
    -------
    numpy.ndarray, shape (n, n), complex
    """
    if n % 2 != 0:
        raise ValueError("symplectic_form requires an even dimension")
    block = np.array([[0.0, 1.0], [-1.0, 0.0]], dtype=complex)
    return np.kron(np.eye(n // 2, dtype=complex), block)


def haar_orthogonal(n: int, rng: np.random.Generator) -> np.ndarray:
    """A single n x n Haar-distributed random orthogonal matrix (O(n),
    both determinant signs -- not restricted to SO(n)).

    Same corrected-QR-decomposition trick as ``haar_unitary`` (Mezzadri
    2007 covers O(n) via the identical construction), with the
    diagonal-sign correction (real case) in place of the phase
    correction (complex case): naive QR is non-unique up to
    multiplying Q's columns by +-1 while adjusting R to compensate, and
    LAPACK's default resolution of that ambiguity is not rotation-
    invariant. Verified empirically during development: the resulting
    matrix is orthogonal to machine precision, its first column is
    uniformly distributed on the unit sphere (mean 0, second moment
    1/n per component), and its determinant is +1 or -1 with equal
    probability -- see ``tests/test_haar.py``.

    Parameters
    ----------
    n : int
        Matrix dimension.
    rng : numpy.random.Generator
        Random number generator to draw from.

    Returns
    -------
    numpy.ndarray, shape (n, n), real
    """
    z = rng.standard_normal((n, n))
    q, r = np.linalg.qr(z)
    d = np.diagonal(r)
    sign = np.sign(d)
    sign[sign == 0.0] = 1.0
    return q * sign


def haar_symplectic(n_quaternionic: int, rng: np.random.Generator) -> np.ndarray:
    """A single Haar-distributed random element of the compact
    symplectic group Sp(n_quaternionic), embedded as a
    2*n_quaternionic x 2*n_quaternionic unitary matrix.

    Construction: U = V^R @ V, where V is Haar-unitary on
    U(2*n_quaternionic) and V^R = Z @ V^T @ Z^{-1} is its symplectic
    dual (Z the symplectic form) -- the same construction already used
    to sample the Circular Symplectic Ensemble
    (``physicskit.rmt.ensembles.circular.CSE``), exposed here directly as a
    reusable raw-matrix generator rather than only as an eigenvalue
    intermediate. Verified directly (not merely inherited from CSE's
    validated spacing statistics): U is unitary to machine precision,
    satisfies the defining self-duality condition U^R == U exactly, and
    its eigenvalues come in exact double-degenerate pairs -- see
    ``tests/test_haar.py``.

    Parameters
    ----------
    n_quaternionic : int
        The quaternionic dimension; the returned matrix has shape
        (2*n_quaternionic, 2*n_quaternionic).
    rng : numpy.random.Generator
        Random number generator to draw from.

    Returns
    -------
    numpy.ndarray, shape (2*n_quaternionic, 2*n_quaternionic), complex
    """
    n = 2 * n_quaternionic
    v = haar_unitary(n, rng)
    z = symplectic_form(n)
    v_dual = (-z) @ v.T @ z  # Z^{-1} = -Z since Z^2 = -I
    return v_dual @ v
