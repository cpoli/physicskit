"""Coverage for physicskit.rmt.utils.tridiagonal's input-validation branches
(each function is exercised through the ensembles that call it, but never
with an invalid n/m that should raise)."""

from __future__ import annotations

import numpy as np
import pytest

from physicskit.rmt.utils.tridiagonal import (
    sample_hermite_beta_eigenvalues,
    sample_hermite_beta_eigenvalues_and_vectors,
    sample_laguerre_beta_eigenvalues,
)


def test_sample_hermite_beta_eigenvalues_rejects_n_below_2():
    with pytest.raises(ValueError):
        sample_hermite_beta_eigenvalues(1, beta=1, rng=np.random.default_rng(0))


def test_sample_hermite_beta_eigenvalues_and_vectors_rejects_n_below_2():
    with pytest.raises(ValueError):
        sample_hermite_beta_eigenvalues_and_vectors(1, beta=1, rng=np.random.default_rng(0))


def test_sample_hermite_beta_eigenvalues_and_vectors_returns_matching_shapes():
    n = 12
    eigenvalues, eigenvectors = sample_hermite_beta_eigenvalues_and_vectors(n, beta=1, rng=np.random.default_rng(1))
    assert eigenvalues.shape == (n,)
    assert eigenvectors.shape == (n, n)
    assert np.all(np.diff(eigenvalues) >= 0)  # ascending
    # eigenvectors should be orthonormal columns
    np.testing.assert_allclose(eigenvectors.T @ eigenvectors, np.eye(n), atol=1e-8)


def test_sample_laguerre_beta_eigenvalues_rejects_n_below_2_and_m_below_n():
    with pytest.raises(ValueError):
        sample_laguerre_beta_eigenvalues(m=10, n=1, beta=2, rng=np.random.default_rng(0))
    with pytest.raises(ValueError):
        sample_laguerre_beta_eigenvalues(m=5, n=10, beta=2, rng=np.random.default_rng(0))
