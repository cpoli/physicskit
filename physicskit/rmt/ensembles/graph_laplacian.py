"""Random graph Laplacians -- spectral graph theory / network transport
ensembles, built on an Erdos-Renyi G(n, p) random graph.

References
----------
F. R. K. Chung, "Spectral Graph Theory", CBMS Regional Conference
Series in Mathematics 92, AMS, 1997 -- normalized Laplacian eigenvalues
lie in [0, 2] for any graph, with 2 attained iff a connected component
is bipartite (used below as ``natural_scale`` for the normalized case).
F. Chung, L. Lu, V. Vu, "Spectra of random graphs with given expected
degrees", Proc. Natl. Acad. Sci. 100 (2003) 6313 -- Erdos-Renyi Laplacian
spectral asymptotics.

Construction: an n x n symmetric 0/1 adjacency matrix A with i.i.d.
Bernoulli(p) entries above the diagonal (zero diagonal -- no self-loops),
mirrored below. Two Laplacian variants:

    combinatorial: L = D - A            (D = diag(degree))
    normalized:    L_sym = I - D^(-1/2) A D^(-1/2)

Isolated vertices (degree 0, possible whenever p is small enough) are
handled by convention: an isolated vertex contributes an exact zero row/
column to A, hence an exact zero eigenvalue to L trivially, and is
assigned D^(-1/2) = 0 for that vertex in L_sym (rather than raising a
division error) -- the standard spectral-graph-theory convention,
matching how an isolated vertex trivially decouples from graph
transport dynamics either way.

Unlike the semicircle/Marchenko-Pastur ensembles, the COMBINATORIAL
Laplacian's spectrum is not naturally centered at 0 (it concentrates
around the mean degree n*p for large n*p, since L = D - A and D
concentrates around n*p -- Chung-Lu-Vu), so ``natural_scale`` performs
no rescaling for that case (dividing by a single scalar cannot both
correctly center and rescale it); the NORMALIZED Laplacian, by contrast,
has an exact, graph-independent bound (eigenvalues in [0, 2]), so no
rescaling is needed there either, for the opposite reason.
"""

import numpy as np

from .base import MatrixEnsemble


def _erdos_renyi_adjacency(n: int, p: float, rng: np.random.Generator) -> np.ndarray:
    upper = np.triu(rng.random((n, n)) < p, k=1)
    return upper.astype(float) + upper.T.astype(float)


class GraphLaplacianEnsemble(MatrixEnsemble):
    """Laplacian eigenvalues of an Erdos-Renyi G(n, p) random graph.

    Parameters
    ----------
    n : int
        Number of vertices.
    p : float
        Edge probability, in (0, 1].
    normalized : bool, optional
        If True (default False), use the normalized Laplacian
        I - D^(-1/2) A D^(-1/2) (eigenvalues in [0, 2]) instead of the
        combinatorial Laplacian D - A.
    """

    def __init__(
        self,
        n: int,
        p: float,
        normalized: bool = False,
        seed: int | np.random.Generator | None = None,
    ) -> None:
        if not 0.0 < p <= 1.0:
            raise ValueError(f"p must be in (0, 1], got {p}")
        super().__init__(n, seed=seed)
        self.p = float(p)
        self.normalized = normalized

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        a = _erdos_renyi_adjacency(self.n, self.p, rng)
        degree = a.sum(axis=1)
        if self.normalized:
            with np.errstate(divide="ignore"):
                d_inv_sqrt = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
            laplacian = np.eye(self.n) - (d_inv_sqrt[:, None] * a) * d_inv_sqrt[None, :]
        else:
            laplacian = np.diag(degree) - a
        laplacian = (laplacian + laplacian.T) / 2.0
        return np.linalg.eigvalsh(laplacian)

    def natural_scale(self) -> float:
        return 1.0
