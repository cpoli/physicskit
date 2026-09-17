"""Coverage for physicskit.rmt.stats.density.empirical_density (previously
never called anywhere in the codebase)."""

from __future__ import annotations

import numpy as np

import physicskit.rmt as rmt
from physicskit.rmt.stats.density import empirical_density


def test_empirical_density_rescaled_and_raw():
    ens = rmt.ensembles.GOE(n=20, seed=0)
    spectrum = ens.sample(n_samples=5)

    centers_r, counts_r = empirical_density(spectrum, bins=20, rescaled=True)
    assert centers_r.shape == counts_r.shape == (20,)
    assert np.all(np.isfinite(counts_r))

    centers_raw, counts_raw = empirical_density(spectrum, bins=20, rescaled=False)
    assert centers_raw.shape == counts_raw.shape == (20,)
    assert np.all(np.isfinite(counts_raw))
    assert not np.allclose(centers_r, centers_raw)  # genuinely different scale
