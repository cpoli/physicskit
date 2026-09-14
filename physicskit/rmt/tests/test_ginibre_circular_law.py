"""Tests for GinOE, GinUE, GinSE and the circular law benchmark.

Circular law convergence is known to be slower than the Hermitian
ensembles' semicircle/Marchenko-Pastur convergence (non-Hermitian
matrices have weaker eigenvalue rigidity) -- confirmed during
development (log-log KS slope ~ -0.37 vs ~ -0.85 to -0.97 for the
semicircle law at comparable N). Thresholds here are calibrated to that
genuinely slower rate, not loosened arbitrarily.
"""

import numpy as np
import pytest
from cache_utils import cached_sample
from scipy.stats import linregress

import physicskit.rmt as rmt

GINIBRE = [
    ("GinOE", rmt.ensembles.GinOE, 1),
    ("GinUE", rmt.ensembles.GinUE, 2),
    ("GinSE", rmt.ensembles.GinSE, 4),
]


@pytest.mark.parametrize("name,cls,beta", GINIBRE)
def test_beta_is_correct(name, cls, beta):
    ens = cls(n=30, seed=0)
    assert ens.beta == beta


@pytest.mark.parametrize("name,cls,beta", GINIBRE)
def test_eigenvalues_are_complex(name, cls, beta):
    ens = cls(n=50, seed=1)
    spectrum = cached_sample(ens, n_samples=3)
    assert np.iscomplexobj(spectrum.eigenvalues)


def test_ginse_produces_2n_eigenvalues():
    # Quaternion Ginibre: n x n quaternion matrix -> 2n complex
    # eigenvalues per sample (genuinely distinct conjugate pairs, NOT
    # deduplicated -- see ensembles/ginibre.py docstring for why this
    # differs from GSE/CSE's true degeneracy).
    ens = rmt.ensembles.GinSE(n=40, seed=2)
    spectrum = cached_sample(ens, n_samples=5)
    assert spectrum.eigenvalues.shape == (5, 80)


def test_ginse_eigenvalues_form_conjugate_pairs_but_are_distinct():
    ens = rmt.ensembles.GinSE(n=30, seed=3)
    spectrum = cached_sample(ens, n_samples=1)
    eigs = spectrum.eigenvalues[0]
    # sort by real then imaginary part; conjugate pairs should be
    # adjacent, with matching real parts and opposite imaginary parts
    order = np.lexsort((eigs.imag, eigs.real))
    sorted_eigs = eigs[order]
    # Rough structural check: for each non-real eigenvalue, its conjugate
    # should also be present in the spectrum.
    non_real = sorted_eigs[np.abs(sorted_eigs.imag) > 1e-6]
    conjugates_present = 0
    for z in non_real:
        if np.any(np.abs(sorted_eigs - np.conj(z)) < 1e-8):
            conjugates_present += 1
    assert conjugates_present == len(non_real)
    # And distinct: no exact duplicate values (unlike GSE/CSE degeneracy)
    assert len(np.unique(np.round(eigs, 10))) == len(eigs)


@pytest.mark.parametrize("name,cls,beta", GINIBRE)
def test_reproducibility(name, cls, beta):
    ens_a = cls(n=60, seed=42)
    ens_b = cls(n=60, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


@pytest.mark.parametrize("name,cls,beta", GINIBRE)
def test_radii_approach_unit_disk_at_large_n(name, cls, beta):
    # NOTE: GinSE embeds an n x n quaternion matrix as a 2n x 2n complex
    # matrix, so its dense eigensolve cost is ~8x that of GinOE/GinUE at
    # the same n -- keep n modest here rather than reusing a size tuned
    # for the real/complex cases.
    n = 400
    ens = cls(n=n, seed=4)
    spectrum = cached_sample(ens, n_samples=1)
    radii = np.abs(spectrum.rescaled.ravel())
    assert radii.max() < 1.35
    assert radii.mean() == pytest.approx(2.0 / 3.0, abs=0.06)  # E[r] for uniform disk = 2/3


@pytest.mark.parametrize("name,cls,beta", GINIBRE)
def test_circular_law_convergence_shrinks_with_n(name, cls, beta):
    # Looped manually (rather than via benchmark.convergence_curve) so
    # each N's sample can be disk-cached individually -- this is the
    # single slowest test in the suite (GinSE's dense 2n x 2n eigensolve
    # dominates), so caching it matters most here.
    benchmark = rmt.validation.CircularLaw()
    n_values = [40, 90, 200, 400]
    results = []
    for n in n_values:
        ens = cls(n=n, seed=17)
        spectrum = cached_sample(ens, n_samples=12)
        results.append(benchmark.validate(spectrum, seed=17))

    ks_stats = np.array([r.ks_statistic for r in results])
    slope = linregress(np.log(n_values), np.log(ks_stats)).slope
    # Genuinely shallower than the Hermitian ensembles' convergence
    # (see module docstring) -- require a real negative trend, not the
    # steeper threshold used for the semicircle/MP tests.
    assert slope < -0.1, f"{name}: expected shrinking KS distance, slope={slope}"
    assert ks_stats[-1] < 0.04


def test_real_ginibre_has_real_eigenvalues_edelman_kostlan_shub():
    # Edelman, Kostlan, Shub (1994): E[# real eigenvalues] ~ sqrt(2n/pi).
    # Checked as an order-of-magnitude sanity check, not a tight
    # quantitative match (this is an expectation over the ensemble, and
    # we only average over a modest number of samples here).
    n = 500
    ens = rmt.ensembles.GinOE(n=n, seed=5)
    spectrum = cached_sample(ens, n_samples=15)
    real_counts = np.sum(np.abs(spectrum.eigenvalues.imag) < 1e-8, axis=1)
    expected = np.sqrt(2 * n / np.pi)
    assert real_counts.mean() == pytest.approx(expected, rel=0.5)


def test_complex_ginibre_has_no_real_eigenvalues_generically():
    ens = rmt.ensembles.GinUE(n=200, seed=6)
    spectrum = cached_sample(ens, n_samples=5)
    real_fraction = np.mean(np.abs(spectrum.eigenvalues.imag) < 1e-8)
    assert real_fraction < 0.02  # essentially none, unlike GinOE
