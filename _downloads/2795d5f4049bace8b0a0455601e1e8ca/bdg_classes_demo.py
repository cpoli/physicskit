r"""
Altland-Zirnbauer Classes: BdG Eigenvalue Symmetry
==================================================

Extending Dyson's threefold way (GOE/GUE/GSE) with particle-hole
(charge-conjugation) symmetry -- appropriate to Bogoliubov-de Gennes
mean-field Hamiltonians of disordered/chaotic superconductors -- yields
four more symmetry classes: D, C, CI, DIII. Each is a
:math:`2N \times 2N` Hermitian matrix built from a block structure such
as (class D, :math:`\beta=2`)

.. math::

    H = \begin{pmatrix} A & B \\ B^\dagger & -A^T \end{pmatrix},
    \qquad A^\dagger = A,\ \ B^T = -B,

with :math:`A` complex Hermitian and :math:`B` complex antisymmetric
(class C instead takes :math:`B` complex symmetric; CI and DIII use an
analogous block built from real symmetric, resp. purely-imaginary
skew-symmetric, :math:`X_1, X_2` blocks -- see
:mod:`physicskit.rmt.ensembles.bdg` for the full construction). This
block structure has a built-in charge-conjugation symmetry
:math:`\mathcal{C} H \mathcal{C}^{-1} = -H`, which forces the spectrum
to be symmetric about :math:`E=0`: eigenvalues come in :math:`\pm
\lambda` pairs (not degenerate copies -- generically distinct).

This example visualizes the Altland-Zirnbauer superconductor classes D,
C, CI, DIII: eigenvalue density, symmetric about E=0 by construction (the
defining particle-hole symmetry).

Note: the literature also reports a *finer* near-zero-energy signature
distinguishing classes C and D (a density dip vs. peak exactly at E=0,
tied to the presence/absence of a Majorana zero mode -- Beenakker and
collaborators). That effect lives at a much smaller energy scale than
this plot resolves and is not something this demo claims to show;
what's demonstrated here is the coarser, unambiguous structural
property (Hermiticity and the exact +-E pairing) validated in
``tests/test_bdg.py``.

References:
A. Altland, M. R. Zirnbauer, Phys. Rev. B 55 (1997) 1142.

Run:
    python examples/paper_replications/bdg_classes_demo.py
"""

import matplotlib.pyplot as plt

import physicskit.rmt as rmt

N = 150
N_SAMPLES = 60
SEED = 2026

fig, axes = plt.subplots(1, 4, figsize=(15, 3.8), sharex=True)

for ax, (label, cls) in zip(
    axes,
    [("D", rmt.ensembles.BdGClassD), ("C", rmt.ensembles.BdGClassC), ("CI", rmt.ensembles.BdGClassCI), ("DIII", rmt.ensembles.BdGClassDIII)],
    strict=True,
):
    ensemble = cls(n=N, seed=SEED)
    spectrum = ensemble.sample(n_samples=N_SAMPLES)
    scale = ensemble.natural_scale()
    eigs = spectrum.eigenvalues.ravel() / scale

    ax.hist(eigs, bins=100, density=True, alpha=0.5, color="steelblue")
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_title(f"Class {label} (beta={cls.beta})")
    ax.set_xlabel("E (rescaled)")
    ax.set_xlim(-1.5, 1.5)

axes[0].set_ylabel("density")
fig.suptitle(
    "Altland-Zirnbauer superconductor classes -- eigenvalue density\n(structural validation only: Hermiticity + exact +-E pairing, not a fitted curve)"
)
fig.tight_layout()
out_path = "bdg_classes_replication.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
