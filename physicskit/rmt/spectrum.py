"""The result of sampling an ensemble: eigenvalues plus the metadata needed
to compare them against theory."""

from dataclasses import dataclass

import numpy as np


@dataclass
class Spectrum:
    """Eigenvalues drawn from a :class:`~physicskit.rmt.ensembles.base.MatrixEnsemble`."""

    eigenvalues: np.ndarray
    """numpy.ndarray, shape (n_samples, n): Raw (unnormalized) eigenvalues,
    one row per independent draw."""

    n: int
    """int: Matrix dimension used to generate this spectrum."""

    beta: float | None
    """float or None: Dyson index of the ensemble that produced this
    spectrum, or None for ensembles outside Dyson's threefold way (e.g.
    ``BuresHallEnsemble``, ``GraphLaplacianEnsemble``)."""

    ensemble: str
    """str: Name of the ensemble class, for provenance/repr purposes."""

    scale: float = 1.0
    """float: Factor by which ``eigenvalues`` should be divided to reach
    the ensemble's standard asymptotic normalization (e.g. semicircle
    support [-2, 2] for the Gaussian ensembles). Set by the ensemble via
    ``natural_scale()``."""

    eigenvectors: np.ndarray | None = None
    """numpy.ndarray, shape (n_samples, m, n), or None: Eigenvectors as
    columns, matching ``eigenvalues`` column-for-column
    (``eigenvectors[s][:, i]`` is the eigenvector for
    ``eigenvalues[s, i]``). Only present when the ensemble was sampled
    with ``sample(return_eigenvectors=True)``; None otherwise. ``m`` is
    usually ``n`` but can differ for ensembles whose eigenvalue count
    isn't exactly ``n`` (see ``n_samples`` elsewhere in this package for
    the analogous eigenvalue-count caveat)."""

    @property
    def flat(self) -> np.ndarray:
        """All eigenvalues from all samples, raveled into one 1-D array.

        Returns
        -------
        numpy.ndarray
            1-D array of all eigenvalues, shape ``(n_samples * eigenvalues.shape[1],)``.
        """
        return self.eigenvalues.ravel()

    @property
    def rescaled(self) -> np.ndarray:
        """Eigenvalues divided by ``scale``.

        This is the array to compare against a theoretical limiting
        distribution.

        Returns
        -------
        numpy.ndarray
            Same shape as ``eigenvalues``.
        """
        return self.eigenvalues / self.scale

    @property
    def n_samples(self) -> int:
        """Number of independent realizations drawn.

        Returns
        -------
        int
        """
        return self.eigenvalues.shape[0]

    def __repr__(self) -> str:
        return f"Spectrum(ensemble={self.ensemble!r}, n={self.n}, beta={self.beta}, n_samples={self.n_samples})"
