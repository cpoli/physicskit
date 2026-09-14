r"""
Bifurcation Diagrams for Continuous Flows: the Duffing Oscillator
======================================================================

Bifurcation diagrams aren't only for discrete maps -- they're just as useful
for continuous, periodically-forced flows, via a *stroboscopic* Poincare
section: sampling the state once per forcing period turns the continuous
flow into an effective discrete map. The flow swept here is the forced,
damped Duffing oscillator,

.. math::

    \ddot{x} + \delta \dot{x} + \alpha x + \beta x^3 = \gamma \cos(\omega t),

with fixed damping :math:`\delta=0.3`, linear stiffness :math:`\alpha=-1`,
cubic stiffness :math:`\beta=1`, and forcing frequency :math:`\omega=1.2`
(so the double-well potential and forcing frequency are held fixed while
only the forcing amplitude :math:`\gamma` is swept). This example sweeps
:math:`\gamma` and, for each value, plots the surviving stroboscopic
samples, using
:func:`physicskit.chaos.visualizers.bifurcation.stroboscopic_bifurcation_sampler` --
the same :func:`~physicskit.chaos.visualizers.bifurcation.plot_bifurcation_diagram`
used for the :doc:`Logistic Map </api/gallery/chaos/maps/plot_logistic_map>`
works here completely unchanged. A companion plot below sweeps the same
:math:`\gamma` range with
:func:`physicskit.chaos.visualizers.divergence.trajectory_divergence` and
:func:`physicskit.chaos.utils.metrics.lyapunov_exponent_from_divergence` to
estimate the largest Lyapunov exponent at each amplitude, turning the
bifurcation diagram's purely qualitative "band looks chaotic" into the
quantitative :math:`\lambda_{\max} > 0` criterion, on the same :math:`\gamma`
axis.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.chaos.systems.continuous import Duffing
from physicskit.chaos.utils.metrics import lyapunov_exponent_from_divergence
from physicskit.chaos.visualizers.bifurcation import (
    plot_bifurcation_diagram,
    stroboscopic_bifurcation_sampler,
)
from physicskit.chaos.visualizers.divergence import trajectory_divergence

# %%
# Build the stroboscopic sampler
# ----------------------------------
# The forcing period is ``2*pi / omega``; sampling the state once per forcing
# period is what turns the continuous flow into an effective discrete map.
omega = 1.2
period = 2.0 * np.pi / omega

sampler = stroboscopic_bifurcation_sampler(
    lambda gamma: Duffing(delta=0.3, alpha=-1.0, beta=1.0, gamma=gamma, omega=omega),
    state0=np.array([0.1, 0.0]),
    sample_period=period,
    n_transient_periods=60,
    n_keep_periods=20,
    component=0,
    dt=0.15,
    rtol=1e-6,
    atol=1e-8,
)

# %%
# Sweep the forcing amplitude
# --------------------------------
# As ``gamma`` increases, watch the single-point regular response give way
# to period-doubled cycles, and eventually a chaotic band -- the same
# period-doubling route to chaos as the Logistic Map, now in a physically
# driven mechanical oscillator. (Looser tolerances than the sampler's
# defaults keep this parameter sweep fast -- it calls the integrator once per
# `gamma` value below.) Each `gamma` value is fully independent of every
# other, so this sweep is also embarrassingly parallel: ``n_jobs`` runs it
# across a thread pool (real speedup here, since `solve_ivp` releases the
# GIL while it integrates), and ``show_progress`` displays a `tqdm` bar.
gamma_values = np.linspace(0.2, 0.5, 150)
fig, ax = plt.subplots(figsize=(9, 5))
plot_bifurcation_diagram(gamma_values, sampler, n_jobs=4, show_progress=True, marker=".", markersize=1.0, ax=ax)
ax.set_xlabel(r"forcing amplitude $\gamma$")
ax.set_ylabel("x (stroboscopic samples)")
ax.set_title("Duffing oscillator: bifurcation diagram vs. forcing amplitude")

plt.show()

# %%
# Quantifying it: the largest Lyapunov exponent, on the same axis
# ---------------------------------------------------------------------
# The bifurcation diagram above only shows *where* the response looks
# irregular; it doesn't say whether that irregularity is genuine exponential
# sensitivity or "just" a very long periodic cycle. Sweeping the same
# `gamma_values` (on a coarser grid, since each point here integrates two
# whole trajectories rather than sampling one) with
# :func:`~physicskit.chaos.visualizers.divergence.trajectory_divergence` and
# fitting the growth rate of two initially nearby trajectories'
# separation gives the largest Lyapunov exponent :math:`\lambda_{\max}`
# directly: it should track the bifurcation diagram closely, crossing zero
# right around where the single point (or handful of stroboscopic points)
# gives way to the dense chaotic band above.
gamma_values_lyap = np.linspace(0.2, 0.5, 50)
lyapunov_exponents = np.empty_like(gamma_values_lyap)
for i, gamma in enumerate(gamma_values_lyap):
    lyap_system = Duffing(delta=0.3, alpha=-1.0, beta=1.0, gamma=float(gamma), omega=omega)
    t_div, delta_t = trajectory_divergence(lyap_system, state0=np.array([0.1, 0.0]), delta_0=1e-8, t_max=150.0, n_points=400, seed=0)
    n_fit = int(0.5 * t_div.size)
    lyapunov_exponents[i] = lyapunov_exponent_from_divergence(t_div[:n_fit], delta_t[:n_fit] / 1e-8)

fig_lyap, ax_lyap = plt.subplots(figsize=(9, 4))
ax_lyap.axhline(0.0, color="black", lw=0.8)
ax_lyap.plot(gamma_values_lyap, lyapunov_exponents, "o-", color="crimson", markersize=3)
ax_lyap.set_xlim(gamma_values[0], gamma_values[-1])
ax_lyap.set_xlabel(r"forcing amplitude $\gamma$")
ax_lyap.set_ylabel(r"largest Lyapunov exponent $\lambda_{max}$")
ax_lyap.set_title(r"$\lambda_{max} > 0$ lines up with the bifurcation diagram's chaotic band")
fig_lyap.tight_layout()

plt.show()
