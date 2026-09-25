"""Universality benchmark: confirms that the semicircle law (and,
optionally, the Wigner surmise) hold across several different entry
distributions, not just Gaussian ones -- a stronger validation of "the
theory is implemented correctly" than fitting a single Gaussian case
ever could be, since it tests the actual universality *claim* of RMT
rather than one specific realization of it.

This module ties together ``physicskit.rmt.ensembles.universality`` (the
entry-distribution-agnostic ensemble) and the already-existing
``WignerSemicircle`` / ``WignerSurmise`` benchmarks -- no new
theoretical curve is needed here, since universality means the SAME
curve applies regardless of entry distribution.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from ..ensembles.universality import (
    GeneralWignerEnsemble,
    exponential_centered_unit_variance,
    rademacher,
    uniform_unit_variance,
)
from .base import ValidationResult
from .wigner_semicircle import WignerSemicircle

EntrySampler = Callable[[np.random.Generator, tuple[int, ...]], np.ndarray]

#: A representative, deliberately varied set of entry distributions:
#: symmetric-bounded (uniform), symmetric-discrete (Rademacher), and
#: asymmetric-unbounded (centered exponential) -- if universality holds
#: across all three, entry-distribution shape/boundedness/symmetry are
#: all ruled out as confounds.
DEFAULT_ENTRY_DISTRIBUTIONS: dict[str, EntrySampler] = {
    "uniform": uniform_unit_variance,
    "rademacher": rademacher,
    "exponential": exponential_centered_unit_variance,
}


@dataclass
class UniversalityResult:
    """Per-distribution validation results plus a summary judgment.

    Attributes
    ----------
    per_distribution : dict of str to ValidationResult
        Semicircle-law validation result for each entry distribution.
    max_ks_statistic : float
        Largest KS statistic across all entry distributions.
    ks_threshold : float
        KS statistic every distribution must stay below for `passed`.
    """

    per_distribution: dict[str, ValidationResult] = field(default_factory=dict)
    max_ks_statistic: float = 0.0
    ks_threshold: float = 0.02

    @property
    def passed(self) -> bool:
        """True if every distribution's KS statistic is below `ks_threshold`."""
        return self.max_ks_statistic < self.ks_threshold

    def __repr__(self):
        lines = [f"  {name}: {result}" for name, result in self.per_distribution.items()]
        verdict = "passed" if self.passed else "failed"
        return "UniversalityResult(\n" + "\n".join(lines) + f"\n  max_ks={self.max_ks_statistic:.5f} ({verdict} at ks_threshold={self.ks_threshold:g})\n)"


def check_universality(
    n: int,
    beta: int,
    n_samples: int = 25,
    seed: int | None = None,
    entry_distributions: dict[str, EntrySampler] | None = None,
    ks_threshold: float = 0.02,
) -> UniversalityResult:
    """Sample ``GeneralWignerEnsemble`` with several entry distributions
    at the given n and beta, validate each against the Wigner semicircle
    law, and return per-distribution results plus a pass/fail summary.

    Parameters
    ----------
    n : int
        Matrix dimension.
    beta : int
        1 or 2 (see ``GeneralWignerEnsemble``).
    n_samples : int
        Number of matrices sampled per entry distribution.
    seed : int or None
        Seed for reproducible sampling.
    entry_distributions : dict or None
        name -> entry_sampler callable; defaults to
        ``DEFAULT_ENTRY_DISTRIBUTIONS``.
    ks_threshold : float
        Each distribution's KS statistic against the semicircle law
        must be below this for the overall check to pass.

    Returns
    -------
    UniversalityResult
        Its ``passed`` property gives the overall verdict against
        `ks_threshold`.
    """
    if entry_distributions is None:
        entry_distributions = DEFAULT_ENTRY_DISTRIBUTIONS

    benchmark = WignerSemicircle()
    per_distribution = {}
    for i, (name, sampler) in enumerate(entry_distributions.items()):
        ensemble_seed = None if seed is None else seed + i
        ensemble = GeneralWignerEnsemble(n=n, entry_sampler=sampler, beta=beta, seed=ensemble_seed)
        spectrum = ensemble.sample(n_samples=n_samples)
        result = benchmark.validate(spectrum, seed=ensemble_seed)
        per_distribution[name] = result

    max_ks = max(r.ks_statistic for r in per_distribution.values())
    return UniversalityResult(per_distribution=per_distribution, max_ks_statistic=max_ks, ks_threshold=ks_threshold)
