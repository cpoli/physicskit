"""Tests for GraphLaplacianEnsemble (Erdos-Renyi combinatorial and
normalized Laplacians)."""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_beta_is_none():
    ens = rmt.ensembles.GraphLaplacianEnsemble(n=30, p=0.3, seed=0)
    assert ens.beta is None


def test_invalid_p_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.GraphLaplacianEnsemble(n=10, p=0.0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.GraphLaplacianEnsemble(n=10, p=1.5, seed=0)


def test_normalized_laplacian_fresh_sample_produces_finite_eigenvalues():
    # A fresh (uncached) direct .sample() call, to actually exercise the
    # normalized=True code path rather than risk a disk-cache hit from a
    # previous run with these same parameters.
    ens = rmt.ensembles.GraphLaplacianEnsemble(n=15, p=0.5, normalized=True, seed=123)
    spectrum = ens.sample(n_samples=2)
    assert np.all(np.isfinite(spectrum.eigenvalues))


def test_reproducibility():
    ens_a = rmt.ensembles.GraphLaplacianEnsemble(n=40, p=0.2, seed=42)
    ens_b = rmt.ensembles.GraphLaplacianEnsemble(n=40, p=0.2, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_combinatorial_laplacian_is_positive_semidefinite_with_zero_mode():
    # L = D - A is always PSD with smallest eigenvalue exactly 0 (the
    # constant vector is always in its kernel, connected or not).
    ens = rmt.ensembles.GraphLaplacianEnsemble(n=50, p=0.3, seed=1)
    spectrum = cached_sample(ens, n_samples=10)
    assert np.all(spectrum.eigenvalues >= -1e-8)
    assert np.abs(spectrum.eigenvalues.min(axis=1)).max() < 1e-8


def test_normalized_laplacian_eigenvalues_bounded_in_zero_two():
    # Exact graph-independent bound (Chung 1997), checked directly.
    ens = rmt.ensembles.GraphLaplacianEnsemble(n=60, p=0.4, normalized=True, seed=2)
    spectrum = cached_sample(ens, n_samples=10)
    assert np.all(spectrum.eigenvalues >= -1e-8)
    assert np.all(spectrum.eigenvalues <= 2.0 + 1e-8)


def test_normalized_laplacian_handles_isolated_vertices():
    # Very small p on a modest n makes isolated vertices likely; should
    # not raise (division-by-zero for degree-0 rows is handled by
    # convention -- see module docstring) and eigenvalues must stay
    # finite and within [0, 2].
    ens = rmt.ensembles.GraphLaplacianEnsemble(n=30, p=0.02, normalized=True, seed=3)
    spectrum = cached_sample(ens, n_samples=10)
    assert np.all(np.isfinite(spectrum.eigenvalues))
    assert np.all(spectrum.eigenvalues >= -1e-8)
    assert np.all(spectrum.eigenvalues <= 2.0 + 1e-8)


def test_combinatorial_laplacian_zero_multiplicity_matches_component_count():
    # Well-known spectral graph theory fact: the multiplicity of the
    # zero eigenvalue equals the number of connected components. Force
    # a disconnected graph (very small p, small n) and check directly
    # via a graph-free connectivity computation (BFS/union-find on the
    # same adjacency matrix used to build L).
    from physicskit.rmt.ensembles.graph_laplacian import _erdos_renyi_adjacency

    rng = np.random.default_rng(4)
    n, p = 25, 0.03
    a = _erdos_renyi_adjacency(n, p, rng)
    degree = a.sum(axis=1)
    laplacian = np.diag(degree) - a
    eigs = np.linalg.eigvalsh(laplacian)
    zero_multiplicity = np.sum(np.abs(eigs) < 1e-8)

    # union-find to count connected components independently
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(n):
        for j in range(i + 1, n):
            if a[i, j] > 0:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj
    n_components = len({find(i) for i in range(n)})

    assert zero_multiplicity == n_components
