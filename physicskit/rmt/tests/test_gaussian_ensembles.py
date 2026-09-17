"""Coverage for physicskit.rmt.ensembles.gaussian.HermiteBetaEnsemble
(previously untested directly): the return_eigenvectors=True guard for
continuum (non-classical) beta values."""

from __future__ import annotations

import numpy as np
import pytest

import physicskit.rmt as rmt


def test_return_eigenvectors_rejects_continuum_beta():
    ens = rmt.ensembles.HermiteBetaEnsemble(n=10, beta=1.5, seed=0)
    with pytest.raises(NotImplementedError):
        ens.sample(n_samples=2, return_eigenvectors=True)


def test_return_eigenvectors_works_for_classical_beta():
    ens = rmt.ensembles.GOE(n=10, seed=0)
    spectrum = ens.sample(n_samples=2, return_eigenvectors=True)
    assert np.all(np.isfinite(spectrum.eigenvalues))
