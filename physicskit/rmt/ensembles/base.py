"""Base class every ensemble in physicskit.rmt implements."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from ..spectrum import Spectrum
from ..utils.random_state import as_generator


class MatrixEnsemble(ABC):
    """A distribution over matrices (or, equivalently, over eigenvalue
    tuples). Subclasses implement ``_sample_eigenvalues`` -- eigenvalues,
    not dense matrices, are the primary product, since several ensembles
    (the Gaussian beta-ensembles via the Dumitriu-Edelman tridiagonal
    model) can produce exact-in-distribution eigenvalues far more
    efficiently than forming and diagonalizing a dense matrix.

    Mirrors the ``scipy.stats`` convention of a distribution object with
    a ``.sample()``/``.rvs()``-style draw method, rather than a bare
    generator function.
    """

    #: Dyson index; set by the subclass (either fixed, e.g. GOE.beta=1,
    #: or passed at construction for continuum-beta ensembles).
    beta: float | None = None

    def __init__(self, n: int, seed: int | np.random.Generator | None = None) -> None:
        """
        Parameters
        ----------
        n : int
            Matrix dimension.
        seed : int, numpy.random.Generator, or None, optional
            Seed for reproducible sampling. See
            :func:`~physicskit.rmt.utils.random_state.as_generator`.
        """
        self.n = n
        self.seed = seed
        self._rng = as_generator(seed)

    @abstractmethod
    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        """Draw one realization's eigenvalues. Must return a 1-D array of
        length ``self.n``.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        numpy.ndarray, shape (n,)
        """
        raise NotImplementedError  # pragma: no cover -- unreachable: ABC blocks
        # instantiating any subclass that doesn't override this abstract method,
        # so this body can never actually run.

    def _sample_eigenvalues_and_vectors(self, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Draw one realization's eigenvalues AND eigenvectors (columns).
        Optional: only ensembles that support eigenvector-level
        statistics (e.g. the inverse participation ratio, see
        ``physicskit.rmt.stats.localization``) override this. The default raises
        NotImplementedError.

        Parameters
        ----------
        rng : numpy.random.Generator

        Returns
        -------
        eigenvalues : numpy.ndarray, shape (n,)
        eigenvectors : numpy.ndarray, shape (m, n)
            Column ``i`` is the eigenvector for ``eigenvalues[i]``.
        """
        raise NotImplementedError(
            f"{type(self).__name__} does not support sample(return_eigenvectors=True) -- no _sample_eigenvalues_and_vectors implementation."
        )

    def sample(self, n_samples: int = 1, return_eigenvectors: bool = False) -> Spectrum:
        """Draw ``n_samples`` independent realizations.

        Parameters
        ----------
        n_samples : int, optional
            Number of independent realizations to draw. Default 1.
        return_eigenvectors : bool, optional
            If True, also draw eigenvectors and attach them to the
            returned ``Spectrum`` (see ``Spectrum.eigenvectors``).
            Substantially more expensive in time and memory (O(n^2) per
            sample instead of O(n)) and only supported by ensembles that
            implement ``_sample_eigenvalues_and_vectors``; raises
            NotImplementedError otherwise. Default False.

        Returns
        -------
        Spectrum
        """
        if return_eigenvectors:
            eig_rows = []
            vec_rows = []
            for _ in range(n_samples):
                eigs, vecs = self._sample_eigenvalues_and_vectors(self._rng)
                eig_rows.append(eigs)
                vec_rows.append(vecs)
            eigenvalues = np.stack(eig_rows)
            eigenvectors: np.ndarray | None = np.stack(vec_rows)
        else:
            rows = [self._sample_eigenvalues(self._rng) for _ in range(n_samples)]
            eigenvalues = np.stack(rows)  # dtype (real or complex) inferred automatically;
            # also accommodates ensembles whose eigenvalue count isn't exactly
            # self.n (e.g. GinSE, which returns 2*n genuinely distinct
            # conjugate-paired eigenvalues per sample -- see ensembles/ginibre.py).
            eigenvectors = None
        return Spectrum(
            eigenvalues=eigenvalues,
            n=self.n,
            beta=self.beta,
            ensemble=type(self).__name__,
            scale=self.natural_scale(),
            eigenvectors=eigenvectors,
        )

    def natural_scale(self) -> float:
        """Factor by which raw eigenvalues should be divided to reach the
        ensemble's standard asymptotic normalization. Default: no rescaling;
        subclasses override where a known theoretical scale applies.

        Returns
        -------
        float
        """
        return 1.0
