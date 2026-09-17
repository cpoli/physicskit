"""Coverage for a handful of never-exercised convenience methods across
physicskit.rmt.validation.*: each Benchmark subclass's theoretical_pdf
(only theoretical_cdf is used internally elsewhere), the sine_kernel and
real_ginibre result types' __repr__, UniversalityResult.__repr__, and the
circular_law/tracy_widom convergence_curve overrides (every existing test
that mentions them deliberately loops manually instead)."""

from __future__ import annotations

import numpy as np

import physicskit.rmt as rmt
from physicskit.rmt.validation.real_ginibre import RealEigenvalueCountResult
from physicskit.rmt.validation.sine_kernel import CorrelationValidationResult
from physicskit.rmt.validation.universality import UniversalityResult


def test_theoretical_pdf_methods_are_finite_and_nonnegative():
    cases = [
        (rmt.validation.MarchenkoPastur(gamma=0.5), np.linspace(0.1, 2.0, 10)),
        (rmt.validation.Wachter(a=1.0, b=1.0), np.linspace(0.1, 0.9, 10)),
        (rmt.validation.WignerSemicircle(), np.linspace(-1.5, 1.5, 10)),
        (rmt.validation.SingleRingTheorem(r_in=0.5, r_out=1.0), np.linspace(0.5, 1.0, 10)),
        (rmt.validation.CircularLaw(), np.linspace(0.1, 0.9, 10)),
    ]
    for benchmark, x in cases:
        pdf = benchmark.theoretical_pdf(x)
        assert np.all(np.isfinite(pdf))
        assert np.all(pdf >= 0.0)


def test_ratio_distribution_and_wigner_surmise_theoretical_pdf():
    ratio = rmt.validation.RatioDistribution(beta=2)
    r = np.linspace(0.0, 5.0, 10)
    assert np.all(np.isfinite(ratio.theoretical_pdf(r)))

    surmise = rmt.validation.WignerSurmise(beta=2)
    s = np.linspace(0.0, 3.0, 10)
    assert np.all(np.isfinite(surmise.theoretical_pdf(s)))


def test_correlation_validation_result_repr():
    result = CorrelationValidationResult(rmse=0.01234, r_max=4.0, n_bins=60)
    assert repr(result) == "CorrelationValidationResult(rmse=0.01234, r_max=4.0, n_bins=60)"


def test_real_eigenvalue_count_result_repr_and_relative_error():
    result = RealEigenvalueCountResult(empirical_mean=8.0, theoretical=10.0, n=100)
    assert result.relative_error == 0.2
    assert repr(result) == "RealEigenvalueCountResult(empirical_mean=8.0000, theoretical=10.0000, n=100, relative_error=0.2000)"


def test_universality_result_repr():
    result = UniversalityResult(per_distribution={"uniform": rmt.validation.base.ValidationResult(0.01, 0.5, 0.02, 50)}, max_ks_statistic=0.01)
    text = repr(result)
    assert text.startswith("UniversalityResult(\n")
    assert "max_ks=0.01000" in text


def test_circular_law_and_tracy_widom_convergence_curve_overrides():
    circ = rmt.validation.CircularLaw()
    results_c = circ.convergence_curve(lambda n, seed: rmt.ensembles.GinUE(n=n, seed=seed), n_values=[20, 40], n_samples=5, seed=0)
    assert [n for n, _ in results_c] == [20, 40]

    tw = rmt.validation.TracyWidom(beta=2)
    results_tw = tw.convergence_curve(lambda n, seed: rmt.ensembles.GUE(n=n, seed=seed), n_values=[20, 40], n_samples=10, seed=0)
    assert [n for n, _ in results_tw] == [20, 40]
