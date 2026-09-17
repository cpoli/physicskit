"""Centralized RNG handling so every ensemble and benchmark in physicskit.rmt
is exactly reproducible from a seed."""

from __future__ import annotations

import numpy as np


def as_generator(seed: int | np.random.Generator | None) -> np.random.Generator:
    """Coerce ``seed`` into a numpy Generator.

    Parameters
    ----------
    seed : int, numpy.random.Generator, or None
        An existing Generator (returned as-is), an integer seed, or
        None (fresh, non-reproducible entropy).

    Returns
    -------
    numpy.random.Generator
    """
    if isinstance(seed, np.random.Generator):
        return seed
    return np.random.default_rng(seed)
