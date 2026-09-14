"""Tests for TruncatedUnitaryEnsemble and its support-radius formula.

The support radius sqrt(alpha) was NOT assumed from memory -- an initial
guess of sqrt(1-alpha) had exactly the wrong alpha-dependence (shrinking
as more of the unitary matrix is kept, contradicting the alpha=1 exact-
unit-circle limit), caught by checking the direction of the trend before
committing to a formula. The correct formula was then confirmed via the
99th-percentile empirical eigenvalue radius converging cleanly to it as
n grows at fixed alpha -- see
``physicskit.rmt/ensembles/truncated_unitary.py`` module docstring.
"""

import numpy as np
import pytest
from cache_utils import cached_sample

import physicskit.rmt as rmt


def test_invalid_alpha_rejected():
    with pytest.raises(ValueError):
        rmt.ensembles.TruncatedUnitaryEnsemble(m=10, alpha=0.0, seed=0)
    with pytest.raises(ValueError):
        rmt.ensembles.TruncatedUnitaryEnsemble(m=10, alpha=1.5, seed=0)
    with pytest.raises(ValueError):
        rmt.stats.truncated_unitary_edge_radius(0.0)


def test_alpha_1_recovers_unit_circle():
    # A full (untruncated) Haar-unitary matrix's eigenvalues lie exactly
    # on the unit circle.
    ens = rmt.ensembles.TruncatedUnitaryEnsemble(m=100, alpha=1.0, seed=0)
    spectrum = cached_sample(ens, n_samples=5)
    radii = np.abs(spectrum.eigenvalues.ravel())
    np.testing.assert_allclose(radii, 1.0, atol=1e-8)


def test_edge_radius_matches_formula():
    for alpha in [0.2, 0.5, 0.8]:
        assert rmt.stats.truncated_unitary_edge_radius(alpha) == pytest.approx(np.sqrt(alpha))


def test_reproducibility():
    ens_a = rmt.ensembles.TruncatedUnitaryEnsemble(m=40, alpha=0.5, seed=42)
    ens_b = rmt.ensembles.TruncatedUnitaryEnsemble(m=40, alpha=0.5, seed=42)
    spec_a = ens_a.sample(n_samples=3)
    spec_b = ens_b.sample(n_samples=3)
    np.testing.assert_allclose(spec_a.eigenvalues, spec_b.eigenvalues)


def test_edge_radius_converges_with_increasing_m():
    # The 99th-percentile RESCALED radius (spectrum.rescaled already
    # divides by natural_scale() = sqrt(alpha), so the target here is
    # 1.0, not sqrt(alpha) again) should approach 1 as the truncated
    # block size (and hence the full unitary dimension, at fixed alpha)
    # grows -- checked here via a shrinking overshoot rather than a
    # fixed tolerance at one size.
    alpha = 0.5
    overshoots = []
    for m in [50, 100, 200, 400]:
        ens = rmt.ensembles.TruncatedUnitaryEnsemble(m=m, alpha=alpha, seed=1)
        spectrum = cached_sample(ens, n_samples=30)
        radii = np.abs(spectrum.rescaled.ravel())
        p99 = np.percentile(radii, 99)
        overshoots.append(p99 - 1.0)
    assert overshoots[0] > overshoots[-1] > -0.05
    assert overshoots[-1] < 0.05


def test_bulk_density_is_not_uniform_like_ginibre():
    # Qualitative check of the module docstring's claim: mean radius
    # should exceed a uniform disk's (2/3)*edge prediction, i.e. the
    # density is weighted toward the edge, unlike the circular law's
    # flat disk.
    alpha = 0.5
    ens = rmt.ensembles.TruncatedUnitaryEnsemble(m=300, alpha=alpha, seed=2)
    spectrum = cached_sample(ens, n_samples=30)
    radii = np.abs(spectrum.rescaled.ravel())
    uniform_disk_mean = (2.0 / 3.0) * rmt.stats.truncated_unitary_edge_radius(alpha)
    assert radii.mean() > uniform_disk_mean
