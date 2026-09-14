"""Tests for the free probability module (Stieltjes/R-transform, free
additive convolution).

The semicircle R-transform (R(w) = w) was derived directly here (solving
w = G(z) for z from the semicircle's own Stieltjes transform), not
recalled, and the generic numerical inversion pipeline
(``r_transform_numerical``) was checked to reproduce it before being
trusted to derive/verify the Marchenko-Pastur R-transform
(R(w) = 1/(1-gamma*w)) against the already-validated exact MP density.
The free additive convolution theorem itself (R_{A+B} = R_A + R_B for
independent, asymptotically free random matrices) is checked directly
against real matrix simulation (independent GOE + Wishart matrices
added together), not just asserted from the individual closed forms.
"""

import numpy as np
import pytest

import physicskit.rmt as rmt


def test_stieltjes_transform_semicircle_matches_empirical():
    ens = rmt.ensembles.GOE(n=2000, seed=0)
    spectrum = ens.sample(n_samples=1)
    eigs = spectrum.rescaled[0]
    for z in [3.0, 5.0, 10.0]:
        empirical = rmt.stats.stieltjes_transform_empirical(eigs, z)
        exact = rmt.stats.stieltjes_transform_semicircle(z)
        assert empirical.real == pytest.approx(exact.real, abs=0.02)


def test_numerical_inversion_recovers_semicircle_r_transform():
    # The pipeline itself, exercised on a Stieltjes transform with a
    # known-exact R-transform (see module docstring for the derivation).
    for w in [0.1, 0.3, 0.6, 0.9]:
        r_numerical = rmt.stats.r_transform_numerical(rmt.stats.stieltjes_transform_semicircle, w, z_bracket=(2.0001, 100.0))
        assert r_numerical == pytest.approx(rmt.stats.r_transform_semicircle(w), rel=1e-4)


def test_numerical_inversion_recovers_marchenko_pastur_r_transform():
    gamma = 0.5
    lo, hi = rmt.stats.mp_support(gamma)

    def stieltjes_mp(z):
        from scipy.integrate import quad

        real, _ = quad(lambda x: rmt.stats.mp_pdf(np.array([x]), gamma)[0] / (z - x), lo, hi, limit=200)
        return real

    for w in [0.1, 0.3, 0.6]:
        r_numerical = rmt.stats.r_transform_numerical(stieltjes_mp, w, z_bracket=(hi + 1e-6, hi + 50.0))
        assert r_numerical == pytest.approx(rmt.stats.r_transform_marchenko_pastur(w, gamma), rel=1e-3)


def test_free_additive_convolution_for_goe_plus_wishart():
    # The genuine content of free probability's additive convolution
    # theorem: for independent (hence asymptotically free) GOE and
    # Wishart matrices A, B, the R-transform of A+B's spectral
    # distribution should equal R_A + R_B -- checked here against DIRECT
    # matrix simulation, not just the two closed forms in isolation.
    rng = np.random.default_rng(0)
    n = 800
    gamma = 0.5
    m = int(n / gamma)

    x = rng.standard_normal((n, n))
    a = (x + x.T) / np.sqrt(2.0) / np.sqrt(n)  # semicircle normalization, variance 1

    y = rng.standard_normal((n, m)) / np.sqrt(m)
    b = y @ y.T  # Marchenko-Pastur normalization at aspect ratio gamma

    c_eigs = np.linalg.eigvalsh(a + b)

    def stieltjes_c(z):
        return rmt.stats.stieltjes_transform_empirical(c_eigs, z)

    z_bracket = (c_eigs.max() + 0.05, c_eigs.max() + 50.0)
    for w in [0.1, 0.2, 0.3]:
        r_c = rmt.stats.r_transform_numerical(stieltjes_c, w, z_bracket)
        r_sum = rmt.stats.r_transform_semicircle(w) + rmt.stats.r_transform_marchenko_pastur(w, gamma)
        assert r_c == pytest.approx(r_sum, rel=0.01)
