r"""
Lamé's thick-walled cylinder: plane-strain finite elements
=============================================================

A gun barrel, a pressure vessel or a hydraulic pipe is a thick cylinder
with pressure :math:`p` inside. Gabriel Lamé (1852) found its stresses in
closed form:

.. math::

    \sigma_{rr} = A - \frac{B}{r^2}, \qquad \sigma_{\theta\theta} = A + \frac{B}{r^2},
    \qquad A = \frac{p a^2}{b^2 - a^2}, \quad B = \frac{p a^2 b^2}{b^2 - a^2}.

The hoop stress peaks at the bore, and a thicker wall helps less and less.
Even an infinitely thick wall only brings the peak hoop stress down to
:math:`p`.

This example solves the same problem numerically with
:func:`~physicskit.fluids.plane_elasticity_solve`, using linear triangles in
plane strain on a quarter of the cross-section, with symmetry conditions on
the cut edges. It compares the displacement and stresses with Lamé's
formulas and shows the finite-element error shrinking under refinement.
"""

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np

from physicskit.fluids import annulus_mesh, lame_thick_cylinder, plane_elasticity_solve, polar_stress, pressure_load

a, b, p, E, nu = 1.0, 2.0, 10.0, 200.0, 0.3


def solve(n_r):
    nodes, tris, bnd = annulus_mesh(a, b, n_r, 2 * n_r)
    fixed = np.zeros((len(nodes), 2), dtype=bool)
    fixed[bnd["theta0"], 1] = True  # u_y = 0 on the x axis
    fixed[bnd["theta_max"], 0] = True  # u_x = 0 on the y axis
    res = plane_elasticity_solve(nodes, tris, E, nu, plane="strain", fixed=fixed, forces=pressure_load(nodes, bnd["inner"], p))
    return nodes, tris, res


nodes, tris, res = solve(20)
r = np.hypot(nodes[:, 0], nodes[:, 1])
u_r = np.sum(res.displacement * nodes, axis=1) / r
srr, stt, _ = polar_stress(res.stress, res.centroids)
r_c = np.hypot(res.centroids[:, 0], res.centroids[:, 1])
rr = np.linspace(a, b, 200)
ex_rr, ex_tt, ex_u = lame_thick_cylinder(rr, a, b, p, E=E, nu=nu, plane="strain")
print(f"bore displacement: FEM {u_r[r == a].mean():.5f}, Lame {ex_u[0]:.5f}")
print(f"peak hoop stress: Lame {ex_tt[0]:.3f} = p (b^2 + a^2)/(b^2 - a^2)")

# %%
# Convergence
# -----------
# Linear triangles give displacements accurate to :math:`O(h^2)` and
# element stresses to :math:`O(h)`.

ns = [4, 8, 16, 32]
u_err, s_err = [], []
for n_r in ns:
    nd, _, rs = solve(n_r)
    rn = np.hypot(nd[:, 0], nd[:, 1])
    u_err.append(np.max(np.abs(np.sum(rs.displacement * nd, axis=1) / rn - lame_thick_cylinder(rn, a, b, p, E=E, nu=nu)[2])))
    _, st, _ = polar_stress(rs.stress, rs.centroids)
    s_err.append(np.sqrt(np.mean((st - lame_thick_cylinder(np.hypot(*rs.centroids.T), a, b, p)[1]) ** 2)))
h = 1.0 / np.array(ns)
print(f"displacement error ~ h^{np.polyfit(np.log(h), np.log(u_err), 1)[0]:.2f}, stress error ~ h^{np.polyfit(np.log(h), np.log(s_err), 1)[0]:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(16, 4.6))
scale = 0.3 / np.max(np.abs(res.displacement))
deformed = nodes + scale * res.displacement
tri = mtri.Triangulation(deformed[:, 0], deformed[:, 1], tris)
tpc = axes[0].tripcolor(tri, facecolors=stt, cmap="inferno")
axes[0].triplot(mtri.Triangulation(nodes[:, 0], nodes[:, 1], tris), color="w", lw=0.15, alpha=0.4)
axes[0].set_aspect("equal")
axes[0].set_title(rf"hoop stress on the deformed quarter (x{scale:.0f})")
fig.colorbar(tpc, ax=axes[0], label=r"$\sigma_{\theta\theta}$")

axes[1].plot(r_c, stt, ".", ms=2, alpha=0.4, color="tab:red", label=r"FEM $\sigma_{\theta\theta}$")
axes[1].plot(r_c, srr, ".", ms=2, alpha=0.4, color="tab:blue", label=r"FEM $\sigma_{rr}$")
axes[1].plot(rr, ex_tt, "k-", lw=1.2, label="Lamé")
axes[1].plot(rr, ex_rr, "k-", lw=1.2)
axes[1].set_xlabel("r")
axes[1].set_ylabel("stress")
axes[1].set_title(f"stresses, p = {p:g}")
axes[1].legend(fontsize=8, markerscale=4)

axes[2].loglog(h, u_err, "o-", label="displacement (max)")
axes[2].loglog(h, s_err, "s-", label="hoop stress (rms)")
axes[2].loglog(h, u_err[0] * (h / h[0]) ** 2, "k--", lw=1, label=r"$h^2$")
axes[2].loglog(h, s_err[0] * (h / h[0]), "k:", lw=1, label=r"$h$")
axes[2].set_xlabel("radial cell size h / (b - a)")
axes[2].set_title("finite-element error")
axes[2].legend(fontsize=8)
fig.tight_layout()
plt.show()
