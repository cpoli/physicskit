"""Coverage for physicskit.rmt.spectrum.Spectrum's simple derived properties
and __repr__ (previously never exercised directly)."""

from __future__ import annotations

import numpy as np

import physicskit.rmt as rmt


def test_flat_ravels_all_samples():
    ens = rmt.ensembles.GOE(n=5, seed=0)
    spectrum = ens.sample(n_samples=3)
    assert spectrum.flat.shape == (15,)
    np.testing.assert_allclose(spectrum.flat, spectrum.eigenvalues.ravel())


def test_n_samples_matches_first_axis():
    ens = rmt.ensembles.GOE(n=5, seed=0)
    spectrum = ens.sample(n_samples=4)
    assert spectrum.n_samples == 4


def test_repr_contains_ensemble_name_and_n_samples():
    ens = rmt.ensembles.GOE(n=5, seed=0)
    spectrum = ens.sample(n_samples=2)
    assert repr(spectrum) == f"Spectrum(ensemble='GOE', n=5, beta={spectrum.beta}, n_samples=2)"
