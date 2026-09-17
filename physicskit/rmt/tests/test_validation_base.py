"""Coverage for physicskit.rmt.validation.base: Benchmark.convergence_curve's
generic (non-overridden) implementation and ValidationResult.__repr__ --
every existing test that mentions convergence_curve deliberately loops
manually instead, so the base implementation itself was never called."""

from __future__ import annotations

import numpy as np

import physicskit.rmt as rmt


def test_convergence_curve_generic_implementation_runs_and_shrinks():
    benchmark = rmt.validation.WignerSemicircle()
    results = benchmark.convergence_curve(lambda n, seed: rmt.ensembles.GOE(n=n, seed=seed), n_values=[20, 60], n_samples=5, seed=0)
    assert [n for n, _ in results] == [20, 60]
    ks_20 = results[0][1].ks_statistic
    ks_60 = results[1][1].ks_statistic
    assert ks_60 < ks_20


def test_validation_result_repr():
    result = rmt.validation.base.ValidationResult(ks_statistic=0.01234, ks_pvalue=0.5, wasserstein_distance=0.005, n_eigenvalues=100)
    assert repr(result) == "ValidationResult(ks_statistic=0.01234, ks_pvalue=0.5, wasserstein_distance=0.00500, n_eigenvalues=100)"
    assert np.isfinite(result.ks_statistic)
