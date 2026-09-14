"""Poisson ensemble -- the integrable-system null model (Berry-Tabor
conjecture).

Reference
---------
M. V. Berry, M. Tabor, "Level clustering in the regular spectrum",
Proc. R. Soc. Lond. A 356 (1977) 375 -- the semiclassical argument that
quantized energy levels of a (generic) classically INTEGRABLE system are,
locally, statistically indistinguishable from an uncorrelated Poisson
point process, in sharp contrast to the level REPULSION seen in
GOE/GUE/GSE for classically CHAOTIC systems (the complementary
Bohigas-Giannoni-Schmit conjecture, Phys. Rev. Lett. 52 (1984) 1).

This is not a matrix ensemble in the sense every other class in this
package is -- there is no underlying random matrix, Dyson index, or
symmetry class here. It is the standard baseline/null model that RMT
level statistics (spacing distribution, ratio statistic, number
variance) are always contrasted against; see
``physicskit.rmt.stats.rigidity.number_variance_poisson``, which until this
ensemble existed had no matching sampler in this package to validate it
against empirically (see ``tests/test_poisson.py``).

Construction: n levels are generated as the partial sums of n i.i.d.
Exponential(1) spacings -- i.e. a single realization of a homogeneous
Poisson point process of UNIT RATE on the positive real line. This is
deliberately NOT "n i.i.d. Uniform(0, n) points" (a common shortcut):
that construction only approaches Poisson-process statistics
asymptotically in the bulk (finite-interval edge effects bias it at
finite n), whereas partial sums of i.i.d. Exponential(1) spacings give
EXACTLY unit-rate Poisson-process statistics at every n -- consecutive
spacings are exactly i.i.d. Exponential(1) by construction, not merely
on average.

Because the process has constant unit rate everywhere (translation
invariant), it needs no unfolding: unlike GOE/Wishart's compact,
nontrivially-shaped limiting density (which spacing/rigidity statistics
must first divide out via a CDF -- see ``physicskit.rmt.stats.unfolding``), the
levels sampled here already have exactly unit mean spacing throughout --
the same "no separate unfolding step needed" situation as the circular
ensembles (``physicskit.rmt.ensembles.circular``), though for the opposite
reason (uniform density on an unbounded line here, vs. exact rotational
invariance on a compact circle there).

``beta`` is left ``None`` (inherited from the base class): Poisson
statistics are not part of Dyson's threefold way at all -- they are the
uncorrelated baseline every beta-ensemble's level repulsion is
contrasted against, NOT a beta -> 0 limit of the Gaussian-tailed Wigner-
surmise family used for GOE/GUE/GSE. That limit is a half-Gaussian
(Rayleigh-shaped) distribution, not the exponential spacing law of a
true Poisson process -- worth knowing if you're tempted to reuse
``physicskit.rmt.validation.WignerSurmise(beta=0)`` or
``physicskit.rmt.validation.RatioDistribution(beta=0)`` here; both would
silently validate against the wrong theoretical curve. The exact
Poisson-process laws instead are: spacing PDF exp(-s) (``scipy.stats.expon``);
and, for the min/max-normalized ratio statistic r in [0, 1] used by
``physicskit.rmt.stats.ratios.ratio_statistics``, PDF 2/(1+r)**2, CDF 2r/(1+r)
-- derived directly here (not taken from a remembered formula) from the
ratio of two i.i.d. Exponential(1) variables, and checked numerically in
the test suite.
"""

import numpy as np

from .base import MatrixEnsemble


class PoissonEnsemble(MatrixEnsemble):
    """n "energy levels" of an integrable system's Berry-Tabor null
    model: a unit-rate homogeneous Poisson point process, realized as
    the partial sums of n i.i.d. Exponential(1) spacings.
    """

    def _sample_eigenvalues(self, rng: np.random.Generator) -> np.ndarray:
        return np.cumsum(rng.exponential(scale=1.0, size=self.n))

    def natural_scale(self) -> float:
        return 1.0
