r"""
SideBySideAnimator: physical space + phase/energy diagnostics, live
==========================================================================

The three animation helpers in ``physicskit.classical.visualizers.animations`` and
``physicskit.classical.visualizers.modal_analysis``, applied to three
already-familiar conservative systems, each paired with a live
diagnostic of its own governing physics: a double pendulum
(:class:`~physicskit.classical.systems.lagrangian.DoublePendulum`,
governed by its Euler-Lagrange equations of motion) shown with its
physical rods next to a growing :math:`(\theta_1, \dot\theta_1)` phase
trace; a precessing Kepler orbit
(:class:`~physicskit.classical.systems.newtonian.KeplerSystem`, whose
post-Newtonian term :math:`c_\mathrm{pn}/r^3` in :math:`V(r)` drives
the precession) shown with its orbital path next to a growing energy
trace; and a non-linear lattice chain
(:class:`~physicskit.classical.systems.chains.FPUTChain`) shown via its
live modal-energy bar chart :math:`E_k(t)`.
"""

# %%
from physicskit.classical.systems.chains import FPUTChain
from physicskit.classical.systems.lagrangian import DoublePendulum
from physicskit.classical.systems.newtonian import KeplerSystem
from physicskit.classical.visualizers.animations import orbit_trace_animation, pendulum_animation
from physicskit.classical.visualizers.modal_analysis import animate_modal_energies


def build_pendulum_animation(stride=500):
    pendulum = DoublePendulum([2.0, 1.0], [0.5, -0.3])
    result = pendulum.integrate((0, 3.0), dt=1e-1, method="implicit_midpoint")
    side_by_side = pendulum_animation(result, pendulum.positions, mode="phase", stride=stride, figsize=(9, 4))
    return side_by_side.build(interval=30)


def build_orbit_animation(stride=20):
    kepler = KeplerSystem.from_orbital_elements(a=1.0, e=0.4)
    result = kepler.integrate((0, 20), dt=1e-1, method="yoshida4")
    side_by_side = orbit_trace_animation(result, mode="energy", stride=stride, figsize=(9, 4))
    return side_by_side.build(interval=30)


def build_modal_energy_animation(stride=10):
    chain = FPUTChain(n=16, beta=0.7, mode=1, amplitude=0.5)
    result = chain.integrate((0, 400), dt=0.1, method="yoshida4")
    return animate_modal_energies(chain, result, stride=stride)


# %%
# Double pendulum: physical rods + growing phase trace
# -------------------------------------------------------------

pendulum_anim = build_pendulum_animation()

# %%
# Precessing Kepler orbit: orbital path + growing energy trace
# ------------------------------------------------------------------

orbit_anim = build_orbit_animation()

# %%
# Lattice chain: live modal-energy bar chart
# -----------------------------------------------

modal_anim = build_modal_energy_animation()
