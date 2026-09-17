r"""
Hohmann's minimum-energy transfer orbit
===========================================

Hohmann (1925) showed that the cheapest way to move between two circular
orbits is a transfer ellipse tangent to both, entered and left with a
single impulsive burn at each tangent point:

.. math::

    a_t = \frac{r_1+r_2}{2}, \qquad
    \Delta v_1 = |v_{\rm t}(r_1)-v_{\rm c}(r_1)|, \qquad
    \Delta v_2 = |v_{\rm c}(r_2)-v_{\rm t}(r_2)|.

:func:`~physicskit.astro.orbital_mechanics.hohmann_transfer` returns
exactly these two burns and the transfer time; this example draws all
three orbits -- the initial and final circular orbits and the transfer
ellipse between them -- for a LEO-to-GEO-like transfer, and checks the
transfer time against half the transfer ellipse's own period.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.orbital_mechanics import (
    hohmann_transfer,
    orbital_period,
    state_from_orbital_elements,
)

# %%
# The transfer: burns and timing
# -----------------------------------
mu = 1.0
r1, r2 = 1.0, 6.0  # a LEO-to-GEO-like ratio, in units where r1 = 1

dv1, dv2, t_transfer = hohmann_transfer(r1, r2, mu)
a_t = (r1 + r2) / 2.0
print(f"burn 1 (leaving r1={r1}):      dv1 = {dv1:.6f}")
print(f"burn 2 (circularizing at r2={r2}): dv2 = {dv2:.6f}")
print(f"transfer time = {t_transfer:.6f}")
print(f"half the transfer ellipse's own period = {orbital_period(a_t, mu) / 2.0:.6f}  (should match transfer time exactly)")

# %%
# The three orbits
# ---------------------
# The initial circular orbit at r1, the transfer ellipse (periapsis at
# r1, apoapsis at r2), and the final circular orbit at r2 -- the
# spacecraft only ever traverses the *lower half* of the transfer
# ellipse, from periapsis to apoapsis, so that half is highlighted.
nu_full = np.linspace(0.0, 2.0 * np.pi, 400)
e_t = (r2 - r1) / (r2 + r1)

circ1 = np.array([state_from_orbital_elements(r1, 0.0, 0.0, 0.0, 0.0, v, mu)[0] for v in nu_full])
circ2 = np.array([state_from_orbital_elements(r2, 0.0, 0.0, 0.0, 0.0, v, mu)[0] for v in nu_full])
transfer_full = np.array([state_from_orbital_elements(a_t, e_t, 0.0, 0.0, 0.0, v, mu)[0] for v in nu_full])
transfer_arc = np.array([state_from_orbital_elements(a_t, e_t, 0.0, 0.0, 0.0, v, mu)[0] for v in np.linspace(0.0, np.pi, 200)])

fig, ax = plt.subplots(figsize=(6.5, 6.5))
ax.plot(circ1[:, 0], circ1[:, 1], color="steelblue", lw=1.5, label=f"initial circular orbit (r={r1})")
ax.plot(circ2[:, 0], circ2[:, 1], color="seagreen", lw=1.5, label=f"final circular orbit (r={r2})")
ax.plot(transfer_full[:, 0], transfer_full[:, 1], "--", color="0.75", lw=1.0)
ax.plot(transfer_arc[:, 0], transfer_arc[:, 1], color="firebrick", lw=2.5, label="transfer ellipse (traversed half)")
ax.plot(r1, 0, "o", color="steelblue", ms=9, zorder=5, label=f"burn 1 (dv={dv1:.3f})")
ax.plot(-r2, 0, "o", color="seagreen", ms=9, zorder=5, label=f"burn 2 (dv={dv2:.3f})")
ax.plot(0, 0, "*", color="orange", ms=16, label="focus")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_title("Hohmann transfer: initial orbit -> transfer ellipse -> final orbit")
ax.set_aspect("equal")
ax.legend(loc="upper right", fontsize=7.5)
fig.tight_layout()

plt.show()
