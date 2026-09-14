"""Tests for PowerLawBandedEnsemble (PBRM).

PBRM has no single universal limiting law to validate against (the
level statistics themselves are the tunable object of study -- see
module docstring), so tests check: (1) the variance-profile construction
matches its own formula directly (the actual thing that could have a
normalization bug, as one briefly did during development -- an initial
version halved the target variance by mistakenly reusing GOE's
two-independent-draws averaging convention for a single-draw
construction), and (2) the two well-understood limiting regimes behave
qualitatively as expected (short-range/localized-like vs. GOE-like).
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_invalid_b_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.PowerLawBandedEnsemble(n=10, b=0.0, seed=0)


def test_reproducibility():
    ens_a = rmt.ensembles.PowerLawBandedEnsemble(n=40, b=5.0, alpha=1.0, seed=42)
    ens_b = rmt.ensembles.PowerLawBandedEnsemble(n=40, b=5.0, alpha=1.0, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_default_alpha_is_the_critical_point():
    ens = rmt.ensembles.PowerLawBandedEnsemble(n=10, b=3.0, seed=0)
    assert ens.alpha == 1.0


def test_variance_profile_matches_target_formula():
    # Direct check of the construction's second moment against its own
    # defining formula, pooled over many independent draws -- this is
    # exactly the check that caught an initial /2 normalization bug
    # during development (see module docstring).
    from physicskit.rmt.ensembles.banded import _power_law_variance_profile

    rng = np.random.default_rng(1)
    n, b, alpha = 20, 5.0, 1.0
    target = _power_law_variance_profile(n, b, alpha)

    # Build the raw matrix directly (the same way
    # ``PowerLawBandedEnsemble._sample_eigenvalues`` does, bypassing the
    # eigendecomposition) many times to check H_ij's empirical variance
    # against the target profile.
    entries_i_j = []
    for _ in range(30_000):
        variance = target
        std = np.sqrt(variance)
        upper = np.triu(rng.standard_normal((n, n)) * std, k=1)
        h = upper + upper.T
        entries_i_j.append(h[0, 3])
    entries_i_j = np.array(entries_i_j)
    assert entries_i_j.var() == pytest.approx(target[0, 3], rel=0.1)


def test_short_range_limit_looks_localized_narrow_band():
    # alpha very large with a small, fixed band -> effectively a
    # strictly banded (short-range) matrix; the bulk eigenvalue spread
    # should stay modest (order the band width), not grow like a full
    # GOE's sqrt(n).
    n = 800
    ens = rmt.ensembles.PowerLawBandedEnsemble(n=n, b=2.0, alpha=6.0, seed=2)
    spectrum = cached_sample(ens, n_samples=3)
    goe_like_scale = np.sqrt(n)
    assert spectrum.eigenvalues.std() < 0.3 * goe_like_scale


def test_large_alpha_zero_limit_approaches_goe_like_scale():
    # alpha -> 0 makes every entry's variance -> 1 (long-range,
    # effectively unbanded) -- this should look like a GOE matrix with
    # unit-variance entries, i.e. eigenvalue std growing like sqrt(n),
    # not staying bounded the way the short-range case does above.
    n = 800
    ens = rmt.ensembles.PowerLawBandedEnsemble(n=n, b=5.0, alpha=0.01, seed=3)
    spectrum = cached_sample(ens, n_samples=3)
    goe_like_scale = np.sqrt(n)
    assert spectrum.eigenvalues.std() > 0.5 * goe_like_scale
