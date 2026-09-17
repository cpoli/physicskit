r"""
Keating-Snaith Moments of the CUE Characteristic Polynomial
=============================================================

For :math:`U` an :math:`n \times n` Haar-random unitary matrix (CUE) and
its characteristic polynomial evaluated on the unit circle,
:math:`Z_n(\theta) = \det(I - U e^{-i\theta})`, the Diaconis-Shahshahani
/ Keating-Snaith formula gives the exact moments (positive integer
:math:`k`; by Haar-measure rotation invariance the value does not
depend on :math:`\theta`):

.. math::

    E\left[|Z_n(\theta)|^{2k}\right] = \prod_{j=0}^{n-1}
    \frac{j!\,(j+2k)!}{(j+k)!^2}.

At :math:`k=1` this product telescopes to the simple closed form
:math:`E[|Z_n|^2] = n+1`. These moments are the random-matrix side of
the Montgomery-Dyson-Keating-Snaith analogy: CUE eigenvalue statistics
are conjectured to model the local statistics of Riemann zeta zeros on
the critical line, and the moments of :math:`Z_n(\theta)` are the
matrix-model analogue of the (still-conjectural) moments of
:math:`\zeta(\tfrac{1}{2}+it)`. This example reproduces the exact
Keating-Snaith moment formula for the CUE characteristic polynomial,
:math:`E[|Z_n(\theta)|^{2k}]`, against direct Monte Carlo simulation of
Haar-random unitary matrices.

References:
H. L. Montgomery, "The pair correlation of zeros of the zeta function",
Proc. Sympos. Pure Math. 24 (1973) 181.
J. P. Keating, N. C. Snaith, Commun. Math. Phys. 214 (2000) 57.

Run:
    python examples/paper_replications/keating_snaith_demo.py
"""

import matplotlib.pyplot as plt
import numpy as np

import physicskit.rmt as rmt

N_VALUES_K1 = np.arange(1, 13)
N_VALUES_K2 = np.arange(1, 9)
SEED = 2026

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

# --- k=1: exact telescoping E[|Z_n|^2] = n + 1 ---
ax = axes[0]
exact_k1 = np.array([rmt.stats.keating_snaith_moment(int(n), 1) for n in N_VALUES_K1])
empirical_k1 = []
for n in N_VALUES_K1:
    ensemble = rmt.ensembles.CUE(n=int(n), seed=SEED)
    spectrum = ensemble.sample(n_samples=20_000)
    empirical_k1.append(rmt.stats.characteristic_polynomial_empirical_moment(spectrum.eigenvalues, k=1))
empirical_k1 = np.array(empirical_k1)

ax.plot(N_VALUES_K1, exact_k1, "k-", lw=2, label=r"exact: $E[|Z_n|^2] = n+1$")
ax.plot(N_VALUES_K1, empirical_k1, "o", color="steelblue", label="Monte Carlo (CUE)")
ax.set_xlabel("n")
ax.set_ylabel(r"$E[|Z_n(\theta)|^2]$")
ax.set_title("k=1 moment")
ax.legend(fontsize=8)

# --- k=2: the general product formula ---
ax = axes[1]
exact_k2 = np.array([rmt.stats.keating_snaith_moment(int(n), 2) for n in N_VALUES_K2])
empirical_k2 = []
for n in N_VALUES_K2:
    ensemble = rmt.ensembles.CUE(n=int(n), seed=SEED + 1)
    spectrum = ensemble.sample(n_samples=150_000)
    empirical_k2.append(rmt.stats.characteristic_polynomial_empirical_moment(spectrum.eigenvalues, k=2))
empirical_k2 = np.array(empirical_k2)

ax.plot(N_VALUES_K2, exact_k2, "k-", lw=2, label=r"exact: $\prod_j j!(j+4)!/(j+2)!^2$")
ax.plot(N_VALUES_K2, empirical_k2, "o", color="indianred", label="Monte Carlo (CUE)")
ax.set_yscale("log")
ax.set_xlabel("n")
ax.set_ylabel(r"$E[|Z_n(\theta)|^4]$")
ax.set_title("k=2 moment (log scale)")
ax.legend(fontsize=8)

# --- Panel 3: the exact moment formula over the full (n, k) grid ---
# Panels 1-2 are each a single fixed-k slice through the exact product
# formula; this shows the whole 2-parameter landscape at once (exact
# values, the same closed form already validated against Monte Carlo
# in the two panels above -- no new sampling needed here).
n_grid_ks = np.arange(1, 13)
k_grid_ks = np.arange(1, 7)
log_moment = np.array([[np.log10(rmt.stats.keating_snaith_moment(int(n), int(k))) for n in n_grid_ks] for k in k_grid_ks])

ax = axes[2]
extent_ks = [n_grid_ks[0] - 0.5, n_grid_ks[-1] + 0.5, k_grid_ks[0] - 0.5, k_grid_ks[-1] + 0.5]
im = ax.imshow(log_moment, aspect="auto", origin="lower", cmap="magma", extent=extent_ks)
ax.set_xlabel("n")
ax.set_ylabel("k")
ax.set_title(r"Exact moment $\log_{10} E[|Z_n(\theta)|^{2k}]$" + "\nover the full (n, k) grid")
fig.colorbar(im, ax=ax, label=r"$\log_{10}$ moment")

fig.suptitle(
    "CUE characteristic polynomial moments: exact Keating-Snaith formula vs. Haar-unitary Monte Carlo",
)
fig.tight_layout()
out_path = "keating_snaith_replication.png"
fig.savefig(out_path, dpi=150)
print(f"Saved {out_path}")
