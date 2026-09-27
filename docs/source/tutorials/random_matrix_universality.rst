Random Matrix Universality: Semicircles, Level Repulsion and the Edge
=====================================================================

Wigner's idea was that the spectrum of a complicated enough Hamiltonian,
such as a heavy nucleus, looks statistically like the spectrum of a
*random* symmetric matrix. The details of the matrix don't matter, only
its symmetry class does. Dyson (1962) showed there are exactly three
such classes, labelled by the index :math:`\beta`: real symmetric
(GOE, :math:`\beta = 1`, time-reversal invariant), complex Hermitian
(GUE, :math:`\beta = 2`, broken time reversal) and quaternion self-dual
(GSE, :math:`\beta = 4`).

This tutorial checks the three universal predictions of that picture
with :mod:`physicskit.rmt`: the global density (the semicircle), the
local correlations (level repulsion) and the extreme eigenvalues
(Tracy-Widom). Random-matrix spectra are dimensionless, so there are no
units to choose.

Sampling an ensemble and the semicircle law
-------------------------------------------

Every ensemble in :mod:`physicskit.rmt.ensembles` returns a
:class:`~physicskit.rmt.spectrum.Spectrum` from ``sample()``, with one row of
eigenvalues per independent matrix. Its ``scale`` attribute is the
factor that brings the spectrum to the standard normalization, which for
the Gaussian ensembles is the semicircle on :math:`[-2, 2]`
(Wigner, *Ann. Math.* 67, 325, 1958):

.. math::

   \rho(x) = \frac{1}{2\pi}\sqrt{4 - x^2}.

.. code-block:: python

   import numpy as np
   from physicskit.rmt import ensembles, stats, validation

   spectrum = ensembles.GOE(n=400, seed=0).sample(n_samples=50)
   print(spectrum.eigenvalues.shape, spectrum.beta, spectrum.scale)
   # (50, 400) 1.0 20.0

   centers, density = stats.empirical_density(spectrum, bins=40)
   print(np.max(np.abs(density - stats.semicircle_pdf(centers))).round(3))
   # 0.009

   print(validation.WignerSemicircle().validate(spectrum, seed=0))
   # ValidationResult(ks_statistic=0.00111, ks_pvalue=1, wasserstein_distance=0.00232, n_eigenvalues=20000)

The histogram agrees with the semicircle to within 0.009 everywhere, and
a Kolmogorov-Smirnov test on all 20 000 eigenvalues cannot tell them
apart. The :mod:`~physicskit.rmt.validation` classes wrap this kind of
comparison: each ties a theoretical law to a test and returns a
:class:`~physicskit.rmt.validation.ValidationResult`.

The semicircle is the weakest of the three predictions, since it doesn't
depend on :math:`\beta`. The signature of each symmetry class is in the
fluctuations.

Level repulsion without unfolding: the spacing ratio
----------------------------------------------------

Eigenvalues of a random matrix avoid each other: the probability of two
levels at distance :math:`s` vanishes like :math:`s^\beta`. Levels of an
integrable system, whose eigenvalues are uncorrelated (Berry and Tabor,
1977), do not repel at all. Measuring spacings usually requires first
*unfolding* the spectrum to unit mean density, which needs the density
itself. The ratio of consecutive spacings (Oganesyan and Huse, 2007)
avoids that step:

.. math::

   \tilde r_i = \frac{\min(s_i, s_{i+1})}{\max(s_i, s_{i+1})}, \qquad s_i = E_{i+1} - E_i .

The local density cancels in the ratio. Its mean is a single number that
separates the four cases (Atas, Bogomolny, Giraud and Roux, *Phys. Rev.
Lett.* 110, 084101, 2013):

.. code-block:: python

   for name, ensemble in [("Poisson", ensembles.PoissonEnsemble), ("GOE", ensembles.GOE),
                          ("GUE", ensembles.GUE), ("GSE", ensembles.GSE)]:
       spectrum = ensemble(n=400, seed=1).sample(n_samples=100)
       r = stats.ratio_statistics(spectrum)
       print(f"{name:<8} beta={spectrum.beta}  <r> = {r.mean():.4f}")

   # Poisson  beta=None  <r> = 0.3868
   # GOE      beta=1.0  <r> = 0.5305
   # GUE      beta=2.0  <r> = 0.5983
   # GSE      beta=4.0  <r> = 0.6745

.. list-table:: Mean spacing ratio, measured and exact
   :header-rows: 1

   * - Ensemble
     - measured
     - large-:math:`n` value (Atas et al. 2013)
   * - Poisson
     - 0.3868
     - :math:`2\ln 2 - 1 = 0.3863`
   * - GOE
     - 0.5305
     - 0.5307
   * - GUE
     - 0.5983
     - 0.5996
   * - GSE
     - 0.6745
     - 0.6744

Each measured value is within 0.002 of its large-:math:`n` limit, and
the four are clearly separated. This is why :math:`\langle\tilde r\rangle`
is the standard diagnostic for whether a physical spectrum is chaotic
(close to GOE) or integrable (close to Poisson). For a Hamiltonian that
you diagonalize yourself, build a
:class:`~physicskit.rmt.spectrum.Spectrum` from its eigenvalues and call
:func:`~physicskit.rmt.stats.ratio_statistics` in exactly the same way.

Nearest-neighbour spacings and the Wigner surmise
-------------------------------------------------

When the mean density is known, as it is here (the semicircle), you can
unfold exactly and look at the spacing distribution itself.
:func:`~physicskit.rmt.stats.nearest_neighbor_spacings` unfolds each
sample through the semicircle CDF and trims the spectrum edges.
Wigner's surmise, the exact result for :math:`2\times 2` matrices,

.. math::

   P_2(s) = \frac{32}{\pi^2} s^2 e^{-4 s^2/\pi},

is accurate to about 1% for large GUE matrices:

.. code-block:: python

   spectrum = ensembles.GUE(n=400, seed=2).sample(n_samples=100)
   s = stats.nearest_neighbor_spacings(spectrum, stats.semicircle_cdf)
   print(len(s), s.mean().round(4), (s < 0.1).mean().round(5), (1 - np.exp(-0.1)).round(5))
   # 31900 0.9998 0.00075 0.09516

   print(validation.WignerSurmise(beta=2).validate(s, seed=0))
   # ValidationResult(ks_statistic=0.00305, ks_pvalue=0.926, wasserstein_distance=0.00190, n_eigenvalues=31900)

The mean spacing is 1, confirming that the unfolding is correct. Only
0.075% of GUE spacings are smaller than a tenth of the mean, compared
with 9.5% (:math:`1 - e^{-0.1}`) for uncorrelated levels. That gap is
level repulsion. The KS test against the surmise gives
:math:`p = 0.93`.

A different ensemble, the same universality: Marchenko-Pastur
--------------------------------------------------------------

Sample covariance matrices :math:`W = X X^T/m`, with :math:`X` an
:math:`n\times m` Gaussian matrix, form the Laguerre (Wishart)
ensembles. Their density is not a semicircle but the Marchenko-Pastur
law (*Mat. Sb.* 72, 507, 1967), supported on
:math:`[(1-\sqrt\gamma)^2, (1+\sqrt\gamma)^2]` with
:math:`\gamma = n/m`. This is the null model for eigenvalues of an
empirical correlation matrix, for example in finance or neuroscience:

.. code-block:: python

   wishart = ensembles.LOE(n=200, m=800, seed=3).sample(n_samples=50)
   lo, hi = stats.mp_support(gamma=200 / 800)
   print(round(lo, 4), round(hi, 4), wishart.rescaled.min().round(3), wishart.rescaled.max().round(3))
   # 0.25 2.25 0.238 2.307

For :math:`\gamma = 1/4` the support is exactly :math:`[1/4, 9/4]`.
Across 10 000 sampled eigenvalues, the extremes overshoot it by only a
few percent. That overshoot is not an error. Its size is set by the
edge statistics, which the next section covers.

The edge: Tracy-Widom
---------------------

The largest eigenvalue of an :math:`n\times n` Gaussian matrix sits at
the spectral edge 2, with fluctuations of order :math:`n^{-2/3}`. That
is much larger than the :math:`n^{-1}` level spacing in the bulk,
because eigenvalues thin out towards the edge.
Rescaled, those fluctuations follow the Tracy-Widom distribution
:math:`F_\beta` (Tracy and Widom, *Commun. Math. Phys.* 159, 151, 1994,
and 177, 727, 1996). That distribution also governs the longest
increasing subsequence of a random permutation and the height
fluctuations of KPZ growth:

.. code-block:: python

   for ensemble, beta in [(ensembles.GOE, 1), (ensembles.GUE, 2)]:
       spectrum = ensemble(n=200, seed=4).sample(n_samples=2000)
       s = (stats.largest_eigenvalues(spectrum) - 2) * stats.tracy_widom_edge_scale(200, beta)
       print(f"beta={beta}: mean {s.mean():+.3f}, std {s.std():.3f}")

   # beta=1: mean -1.254, std 1.299
   # beta=2: mean -1.818, std 0.887

The exact values are :math:`\langle s\rangle = -1.2065`,
:math:`\sigma = 1.2680` for :math:`\beta = 1` and
:math:`\langle s\rangle = -1.7711`, :math:`\sigma = 0.9018` for
:math:`\beta = 2`. The remaining differences of a few percent are the
expected finite-:math:`n` corrections, and they shrink as :math:`n`
grows. :func:`~physicskit.rmt.stats.tracy_widom_cdf` gives the full
distribution for a direct comparison, and
:class:`~physicskit.rmt.validation.TracyWidom` wraps it as a validation
test.

Where to go next
----------------

- :func:`~physicskit.rmt.validation.check_universality` repeats the
  semicircle test for Wigner matrices whose entries are *not* Gaussian
  (uniform, Rademacher and others). The law holds for all of them, which
  is the universality claim itself.
- :mod:`physicskit.rmt.ensembles` also contains circular (COE/CUE/CSE),
  non-Hermitian (Ginibre), chiral, Bogoliubov-de Gennes, SYK and
  crossover ensembles, all returning the same
  :class:`~physicskit.rmt.spectrum.Spectrum`.
- The rmt :doc:`history </history/rmt_breakthroughs>` follows these
  results from Wigner's nuclei to quantum chaos.
