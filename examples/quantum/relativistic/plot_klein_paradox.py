r"""
Klein's paradox: relativistic particles at a tall potential step
================================================================

In 1929 Oskar Klein found that a Dirac electron hitting a step
:math:`V_0` higher than :math:`E + mc^2` is partly transmitted, however
tall the step. The transmitted wave is a negative-energy state: the step
bends the antiparticle continuum down to the particle's energy.
:func:`~physicskit.quantum.chapters.relativistic.dirac_step_scattering`
and :func:`~physicskit.quantum.chapters.relativistic.klein_gordon_step_scattering`
show the three regimes: ordinary transmission, total reflection, and the
Klein zone. There, fermions have :math:`R < 1` but spin-0 bosons have
:math:`R > 1`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.relativistic import dirac_step_scattering, klein_gordon_step_scattering

m, E = 1.0, 2.0
V0 = np.linspace(0, 8, 1601)
V0 = V0[np.abs(V0 - 2 * E) > 0.02]  # skip the KG pole at V0 = 2E

R_d, T_d = dirac_step_scattering(E, V0, m)
R_kg, T_kg = klein_gordon_step_scattering(E, V0, m)
R_naive, _ = dirac_step_scattering(E, V0, m, group_velocity=False)

# %%
# Reflection coefficient across the three regimes
# -------------------------------------------------

fig, ax = plt.subplots(figsize=(10, 5))
ax.axvspan(E - m, E + m, color="0.9", label=r"$|E - V_0| < m$: total reflection")
ax.axvspan(E + m, V0.max(), color="C2", alpha=0.08, label=r"Klein zone $V_0 > E + m$")
ax.plot(V0, R_d, lw=2, label="Dirac (spin 1/2)")
ax.plot(V0, R_kg, lw=2, label="Klein-Gordon (spin 0)")
ax.plot(V0, R_naive, ":", color="C0", label="Dirac, naive $q > 0$ choice")
ax.axhline(1, color="k", lw=0.5)
ax.set_ylim(0, 4)
ax.set_xlabel(r"step height $V_0 / mc^2$")
ax.set_ylabel("reflection coefficient R")
ax.set_title(rf"Scattering off a potential step at $E = {E}\,mc^2$")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()

# %%
# Transmission that never switches off
# -------------------------------------
#
# As :math:`V_0 \to \infty` the Dirac transmission tends to
# :math:`4\kappa_\infty/(1+\kappa_\infty)^2` with
# :math:`\kappa_\infty = \sqrt{(E+m)/(E-m)}`: a finite fraction of the beam
# always gets through, the transmitted current being carried by
# antiparticles created at the step.

V_big = np.logspace(0.5, 4, 200)
_, T_big = dirac_step_scattering(E, V_big, m)
kinf = np.sqrt((E + m) / (E - m))
fig2, ax2 = plt.subplots(figsize=(8, 4))
ax2.semilogx(V_big, T_big, lw=2, label="Dirac transmission T")
ax2.axhline(4 * kinf / (1 + kinf) ** 2, color="k", ls="--", label=r"$4\kappa_\infty/(1+\kappa_\infty)^2$")
ax2.set_xlabel(r"$V_0 / mc^2$")
ax2.set_ylabel("T")
ax2.set_title("Klein tunneling persists for arbitrarily tall steps")
ax2.legend()
fig2.tight_layout()

print(f"R + T = 1 everywhere: Dirac {np.allclose(R_d + T_d, 1)}, Klein-Gordon {np.allclose(R_kg + T_kg, 1)}")
