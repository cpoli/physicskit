"""Tests for CUE characteristic polynomial moments (Keating-Snaith).

The product formula was verified numerically against direct Monte Carlo
(Haar-random unitary matrices, evaluated at theta=0) at several (n, k)
pairs before being trusted -- see
``physicskit.rmt/stats/characteristic_polynomial.py`` module docstring.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_n1_k1_matches_hand_derivation():
    # 1x1 case: Z_1 = 1 - e^{i(phi-theta)}, E[|Z_1|^2] = 2 - 2*E[cos] = 2
    # exactly (uniform phi over a full period).
    assert rmt.stats.keating_snaith_moment(1, 1) == pytest.approx(2.0)


def test_k1_special_case_is_n_plus_1():
    # Noticed and checked during development: k=1 telescopes to n+1.
    for n in [1, 2, 3, 4, 8]:
        assert rmt.stats.keating_snaith_moment(n, 1) == pytest.approx(n + 1)


def test_rejects_non_positive_k():
    with pytest.raises(ValueError):
        rmt.stats.keating_snaith_moment(5, 0)
    with pytest.raises(ValueError):
        rmt.stats.characteristic_polynomial_empirical_moment(np.zeros((1, 3)), 0)


@pytest.mark.parametrize("n,k", [(2, 1), (3, 1), (4, 1), (2, 2)])
def test_moment_matches_empirical_cue_simulation(n, k):
    ens = rmt.ensembles.CUE(n=n, seed=0)
    spectrum = cached_sample(ens, n_samples=20000)
    theory = rmt.stats.keating_snaith_moment(n, k)
    empirical = rmt.stats.characteristic_polynomial_empirical_moment(spectrum.eigenvalues, k)
    assert empirical == pytest.approx(theory, rel=0.05)


def test_moments_grow_with_k():
    # Higher moments of a nonnegative random variable are larger (given
    # the variable isn't a.s. constant, which |Z_n| generically isn't).
    n = 4
    m1 = rmt.stats.keating_snaith_moment(n, 1)
    m2 = rmt.stats.keating_snaith_moment(n, 2)
    assert m2 > m1
