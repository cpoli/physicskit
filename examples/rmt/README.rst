Examples
========

Runnable, self-contained scripts reproducing the key breakthroughs behind
``physicskit.rmt``, organized around Dyson's threefold way: each script
reproduces a classic random matrix theory result to numerical precision
against its published closed-form law or reference (see
:doc:`/history/rmt_breakthroughs` for the full chronology).

Each script in this gallery is self-contained and can be run directly with
``python examples/rmt/<section>/<script>.py``. Every script also carries an
RST module docstring as its title/description and uses ``# %%`` markers to
split narrative text from code, which is exactly what Sphinx-Gallery renders
into the pages below -- the script *is* the source of truth for what you see,
not a copy of it.

Sections
--------

- **edge_statistics** -- the Tracy-Widom soft-edge laws for the largest
  eigenvalue of the GOE, GUE, and GSE.
- **gaussian_ensembles** -- bulk spectra of Hermitian random matrices:
  Wigner's semicircle law, the Two-Body Random Ensemble's Gaussian (not
  semicircular) many-body density of states, Anderson localization in
  power-law banded matrices, the Altland-Zirnbauer Bogoliubov-de Gennes
  classes' eigenvalue symmetry, and the Dumitriu-Edelman tridiagonal model
  that makes continuum-beta simulation possible at all.
- **non_hermitian** -- complex spectra: Ginibre's circular law, the
  Feinberg-Zee single ring theorem, and Bender-Boettcher PT-symmetry
  breaking in pseudo-Hermitian ensembles.
- **spacings** -- local and long-range spectral correlations: the Wigner
  surmise, Dyson's circular ensembles, the Keating-Snaith moments of the CUE
  characteristic polynomial, the Poisson (Berry-Tabor) integrable baseline,
  the Pandey-Mehta GOE-GUE crossover, the Bohigas-Giannoni-Schmit spectral
  rigidity and universality, and the Wigner-Dyson many-body level
  statistics of Kitaev's SYK model.
- **wishart** -- covariance-type ensembles: Wishart's original
  sample-covariance construction, the Marchenko-Pastur law, the Wachter law
  for Jacobi (MANOVA) ensembles, and Page's conjecture for the average
  entanglement entropy of a random quantum state.
