"""Coverage for physicskit.rmt.utils.random_state.as_generator: the
existing-Generator passthrough branch (every ensemble test seeds with a
plain int, so this was never exercised)."""

from __future__ import annotations

import numpy as np

from physicskit.rmt.utils.random_state import as_generator


def test_as_generator_returns_an_existing_generator_unchanged():
    rng = np.random.default_rng(42)
    assert as_generator(rng) is rng


def test_as_generator_coerces_an_int_seed():
    rng = as_generator(7)
    assert isinstance(rng, np.random.Generator)
