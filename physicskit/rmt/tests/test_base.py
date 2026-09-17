"""Coverage for physicskit.rmt.ensembles.base.MatrixEnsemble's default
natural_scale (every concrete ensemble in this codebase overrides it, so a
minimal dummy subclass is needed to exercise the base implementation)."""

from __future__ import annotations

import numpy as np

from physicskit.rmt.ensembles.base import MatrixEnsemble


class _DummyEnsemble(MatrixEnsemble):
    """Implements only the required abstract method; relies on the base
    class's default natural_scale (1.0, no rescaling)."""

    def _sample_eigenvalues(self, rng):
        return rng.standard_normal(self.n)


def test_default_natural_scale_is_one():
    ens = _DummyEnsemble(n=4, seed=0)
    assert ens.natural_scale() == 1.0


def test_sample_uses_default_natural_scale_unscaled():
    ens = _DummyEnsemble(n=4, seed=0)
    spectrum = ens.sample(n_samples=3)
    assert spectrum.scale == 1.0
    assert np.all(np.isfinite(spectrum.eigenvalues))
