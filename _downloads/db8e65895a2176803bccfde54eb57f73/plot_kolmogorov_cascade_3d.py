r"""
Kolmogorov's 1941 cascade in forced 3D turbulence
=================================================

Kolmogorov's picture is three-dimensional: energy fed in at large scales
is handed down a chain of ever smaller eddies, at a rate
:math:`\varepsilon` independent of scale, until viscosity removes it. In
that inertial range the spectrum can only depend on :math:`\varepsilon`
and :math:`k`, so

.. math::

    E(k) = C_K\,\varepsilon^{2/3}k^{-5/3}.

:class:`~physicskit.fluids.systems.turbulence.ForcedTurbulence3D` solves the
incompressible Navier-Stokes equations pseudo-spectrally on a :math:`64^3`
periodic cube, injecting energy at exactly :math:`\varepsilon = 1` into the
modes :math:`k < 2.5` and removing it with a :math:`k^4` hyperviscosity near
the grid scale. Once the flow is statistically steady, the dissipation
balances the injection and the time-averaged spectrum shows a short
:math:`k^{-5/3}` range. At this resolution the range spans only about
:math:`3 \le k \le 10`, and the hyperviscosity adds a small "bottleneck"
bump before the dissipation range.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.fluids.systems.turbulence import ForcedTurbulence3D

flow = ForcedTurbulence3D(n=64, epsilon=1.0, seed=0)
out = flow.run(t_max=12.0, dt=0.01, t_average=6.0)
k, E = out["k"], out["E"]

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

# %%
# Energy budget: injection balanced by dissipation
# ------------------------------------------------
axes[0].plot(out["t"], out["energy"], color="navy", label="kinetic energy")
axes[0].plot(out["t"], out["dissipation"], color="crimson", label=r"dissipation rate $D$")
axes[0].axhline(flow.epsilon, color="crimson", ls=":", label=r"injection $\varepsilon$")
axes[0].axvspan(out["t"][-1] - 6.0, out["t"][-1], color="gray", alpha=0.15, label="averaging window")
axes[0].set_xlabel("t")
axes[0].set_title("Statistically steady state")
axes[0].legend(fontsize=8)

# %%
# The spectrum and its compensated form
# -------------------------------------
inertial = (k >= 3) & (k <= 10)
slope = np.polyfit(np.log(k[inertial]), np.log(E[inertial]), 1)[0]
axes[1].loglog(k[1:], E[1:], color="navy", label="measured")
kk = k[inertial]
axes[1].loglog(kk, 2.0 * E[3] * (kk / 3) ** (-5 / 3), "k--", label=r"$k^{-5/3}$")
axes[1].set_xlabel("k")
axes[1].set_ylabel("E(k)")
axes[1].set_title(f"Fitted slope {slope:.2f} for 3 <= k <= 10")
axes[1].legend()
compensated = E[1:] * k[1:] ** (5 / 3) / flow.epsilon ** (2 / 3)
axes[2].semilogx(k[1:], compensated, color="navy")
axes[2].axvspan(3, 10, color="gold", alpha=0.3, label="inertial range")
axes[2].set_xlabel("k")
axes[2].set_ylabel(r"$E(k)\,k^{5/3}\varepsilon^{-2/3}$")
axes[2].set_title("Compensated spectrum: flat where K41 holds")
axes[2].legend()
plt.tight_layout()
plt.show()
D_mean = out["dissipation"][out["t"] > out["t"][-1] - 6.0].mean()
print(f"inertial-range slope {slope:.3f} (K41 {-5 / 3:.3f}); mean dissipation {D_mean:.3f} (injection {flow.epsilon})")

# %%
# Check
# -----
assert abs(slope + 5 / 3) < 0.15
assert abs(D_mean - flow.epsilon) < 0.1
