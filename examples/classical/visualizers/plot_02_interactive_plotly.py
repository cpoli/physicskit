r"""
Interactive Plotly visualizers: SO(3) momentum sphere and orbit
======================================================================

Unlike the static Matplotlib plots elsewhere in
``physicskit.classical.visualizers``, these are draggable/zoomable in a
live Python session or a browser -- useful for directly inspecting how
close a rigid-body trajectory passes to the unstable intermediate-axis
pole, or for zooming into a precessing orbit's perihelion. The two
systems are governed by the equations already introduced elsewhere in
the gallery:
:class:`~physicskit.classical.systems.rotations.EulerTop`'s torque-free
Euler equations,
:math:`I_i\dot\omega_i = (I_j - I_k)\,\omega_j\omega_k` (cyclic in
:math:`i,j,k`) for the body-frame angular velocity (see
:doc:`/api/gallery/classical/rotations/plot_01_dzhanibekov`), and
:class:`~physicskit.classical.systems.newtonian.KeplerSystem`'s
separable Hamiltonian :math:`H = |p|^2/(2\mu) + V(r)` with the
post-Newtonian perturbation :math:`c_\mathrm{pn}/r^3` in :math:`V(r)`
that drives the perihelion precession (see
:doc:`/api/gallery/classical/newtonian/plot_02_kepler_precession`).
"""

# %%
from physicskit.classical.core.base_system import SimulationResult
from physicskit.classical.systems.newtonian import KeplerSystem
from physicskit.classical.systems.rotations import EulerTop
from physicskit.classical.visualizers.interactive import interactive_orbit, interactive_so3_momentum_sphere

# The integrations below use a fine dt for numerical accuracy, but an
# interactive plot of every single step (potentially hundreds of
# thousands of points) is both unnecessarily large and sluggish to pan
# and zoom -- downsample with a stride to a few hundred points, plenty
# to see the geometry clearly.


def _stride(result: SimulationResult, stride: int) -> SimulationResult:
    idx = list(range(0, len(result.t), stride))
    if idx[-1] != len(result.t) - 1:
        idx.append(len(result.t) - 1)
    q = result.q[idx] if result.q is not None else None
    p = result.p[idx] if result.p is not None else None
    return SimulationResult(t=result.t[idx], y=result.y[idx], q=q, p=p)


# %%
# SO(3) momentum sphere for a tumbling (intermediate-axis) Euler top
# --------------------------------------------------------------------------

top = EulerTop([0.01, 1.0, 0.01], I1=1.0, I2=2.0, I3=3.0)
result_top = _stride(top.integrate((0, 100), dt=1e-1, method="implicit_midpoint"), stride=200)
fig_sphere = interactive_so3_momentum_sphere(result_top.y[:, :3], I1=1.0, I2=2.0, I3=3.0, title="EulerTop: tumbling about the intermediate axis")

# %%
# A precessing Kepler orbit with its LRL vector rotating
# ------------------------------------------------------------

kepler = KeplerSystem.from_orbital_elements(a=1.0, e=0.3, c_pn=0.02)
result_orbit = _stride(kepler.integrate((0, 300), dt=1e-3, method="yoshida4"), stride=300)
fig_orbit = interactive_orbit(result_orbit, system=kepler, title="Precessing Kepler orbit (c_pn=0.02)")
