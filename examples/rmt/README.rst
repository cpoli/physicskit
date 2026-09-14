Examples
========

Paper-replication scripts for ``physicskit.rmt``, organized around Dyson's
threefold way: each script reproduces a classic random matrix theory result
to numerical precision against its published closed-form law or reference.

Every script is real Python you can run directly
(``python examples/rmt/paper_replications/<script>.py``) and is also executed
live at documentation-build time to produce the gallery pages below -- the
script *is* the source of truth for what you see, not a copy of it.

Replications
------------

- **Wigner-Dyson statistics** -- Wigner's semicircle law for the bulk
  spectral density, the Wigner surmise for nearest-neighbor level spacing,
  Tracy-Widom soft-edge laws, spectral rigidity and universality, the
  Pandey-Mehta GOE-GUE crossover, and the Poisson (Berry-Tabor) baseline
  for integrable, non-chaotic spectra.
- **Beyond GOE/GUE/GSE** -- Dyson's circular ensembles (COE, CUE, CSE) and
  the Keating-Snaith moments of the CUE characteristic polynomial, the
  Marchenko-Pastur law for Wishart covariance matrices (and Wishart's
  original finite-sample data-matrix construction), the Wachter law for
  Jacobi ensembles, and Ginibre's circular law for non-Hermitian matrices.
- **Symmetry classes, disorder, and non-Hermiticity** -- the
  Altland-Zirnbauer superconductor classes' Bogoliubov-de Gennes eigenvalue
  symmetry, the Anderson localization transition in power-law banded
  matrices, and PT-symmetry breaking in pseudo-Hermitian ensembles.
- **Many-body and computational random matrix theory** -- the Two-Body
  Random Ensemble's Gaussian (not semicircular) many-body density of
  states, Kitaev's SYK model and its Wigner-Dyson many-body level
  statistics, Page's conjecture for the average entanglement entropy of a
  random quantum state, and the Dumitriu-Edelman tridiagonal model that
  makes continuum-beta simulation possible at all.
