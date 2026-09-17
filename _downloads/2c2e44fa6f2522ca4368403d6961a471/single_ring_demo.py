r"""
The Single Ring Theorem
=======================

For a bi-unitarily-invariant non-Hermitian matrix :math:`M = U \,
\mathrm{diag}(s_1, \dots, s_n)\, V`, built from independent Haar-random
unitary matrices :math:`U, V` and singular values :math:`s_i`, the
single ring theorem states that as :math:`n \to \infty` the eigenvalues
fill the annulus :math:`r_{\text{in}} \leq |z| \leq r_{\text{out}}`,
with radii fixed by only the first two moments of the limiting
singular-value distribution:

.. math::

    r_{\text{out}} = \sqrt{\langle s^2 \rangle}, \qquad
    r_{\text{in}} = \frac{1}{\sqrt{\langle 1/s^2 \rangle}}.

This example builds a non-Hermitian generalization of the Wishart
ensemble by drawing the squared singular values exactly from the
beta-Laguerre (Marchenko-Pastur) distribution at aspect ratio
:math:`\gamma = n/m < 1`, then scrambling them by independent Haar
:math:`U, V`. Because the Marchenko-Pastur distribution has the exact
moments :math:`E[X]=1` and :math:`E[1/X] = 1/(1-\gamma)`, the ring radii
here reduce to the closed form

.. math::

    r_{\text{out}} = 1, \qquad r_{\text{in}} = \sqrt{1-\gamma},

a genuine ring (:math:`r_{\text{in}} > 0`, unlike the filled disk of the
ordinary circular law) whenever :math:`\gamma < 1`. This example
reproduces the single ring theorem for a non-Hermitian Wishart-type
ensemble: eigenvalue scatter in the complex plane fills a genuine
annulus (not a disk), with exact ring radii from the Marchenko-Pastur
distribution's moments.

References:
J. Feinberg, A. Zee, Nucl. Phys. B 504 (1997) 579.
A. Guionnet, M. Krishnapur, O. Zeitouni, Ann. of Math. 174 (2011) 1189.

Run:
    python examples/paper_replications/single_ring_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

import physicskit.rmt as rmt

N = 300
SEED = 2026

fig, axes = plt.subplots(2, 3, figsize=(13, 8))
theta_circle = np.linspace(0, 2 * np.pi, 300)
r_grid = np.linspace(0, 1.2, 300)

GAMMAS = [0.9, 0.5, 0.2]

for col, gamma in enumerate(GAMMAS):
    m = int(N / gamma)
    ensemble = rmt.ensembles.NonHermitianWishartEnsemble(n=N, m=m, beta=2, seed=SEED)
    r_in, r_out = rmt.stats.single_ring_radii_wishart_theory(ensemble.gamma)

    spectrum = ensemble.sample(n_samples=1)
    eigs = spectrum.rescaled[0]

    ax = axes[0, col]
    ax.scatter(eigs.real, eigs.imag, s=4, alpha=0.6, color="darkorange")
    ax.plot(r_out * np.cos(theta_circle), r_out * np.sin(theta_circle), "k-", lw=1.5)
    ax.plot(r_in * np.cos(theta_circle), r_in * np.sin(theta_circle), "k-", lw=1.5)
    ax.set_title(f"gamma={gamma}\nr_in={r_in:.3f}, r_out={r_out:.3f}")
    ax.set_aspect("equal")
    ax.set_xlabel("Re")

    spectrum_many = ensemble.sample(n_samples=15)
    radii = np.abs(spectrum_many.rescaled.ravel())
    benchmark = rmt.validation.SingleRingTheorem(r_in=r_in, r_out=r_out)
    result = benchmark.validate(spectrum_many, seed=SEED)

    ax2 = axes[1, col]
    ax2.hist(radii, bins=50, density=True, alpha=0.5, color="steelblue", label="empirical")
    ax2.plot(
        r_grid,
        rmt.stats.annulus_radial_pdf(r_grid, r_in, r_out),
        "k-",
        lw=2,
        label="annulus theory",
    )
    ax2.set_title(f"radial density\nKS={result.ks_statistic:.4f}")
    ax2.set_xlabel("|eigenvalue|")
    ax2.legend(fontsize=8)

axes[0, 0].set_ylabel("Im")
axes[1, 0].set_ylabel("density")
fig.suptitle(f"Single ring theorem (non-Hermitian Wishart) -- N={N}")
fig.tight_layout()
out_path = "single_ring_replication.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
