r"""
The Ehrenfest urn: reversibility, recurrence, and the arrow of time
=========================================================================

Two objections were raised against Boltzmann's H-theorem soon after he
proposed it: Loschmidt argued that time-reversible microscopic dynamics
cannot produce genuinely irreversible macroscopic behavior, and Zermelo
noted that any bounded mechanical system must (by Poincare recurrence)
eventually return arbitrarily close to its initial state. The Ehrenfests'
1907 urn model resolves both at once, in the simplest possible setting.

:math:`N` labeled balls are split between two boxes; at each discrete time
step, one of the :math:`N` balls, chosen uniformly at random, is moved to
the *other* box. This single rule is exactly reversible (running it
backward is itself a valid Ehrenfest step) and, since the system has only
finitely many configurations, must eventually recur exactly. Writing
:math:`n` for the number of balls in the left box, the Boltzmann entropy of
that macrostate is

.. math::

    S / k_B = \ln W(n), \qquad W(n) = \binom{N}{n},

since :math:`W(n)` counts the ball-labelings compatible with occupancy
:math:`n`; :math:`W(n)`, and hence the entropy, is maximized at the
balanced occupancy :math:`n = N/2` and falls off sharply toward either
extreme.

This example starts every one of :math:`N = 500` balls in the left box
(:math:`n = 0`, the state of minimum :math:`W` and hence minimum entropy --
a maximally non-equilibrium state) and watches the occupancy relax toward
:math:`N/2` while the entropy climbs monotonically toward its maximum, even
though the underlying per-ball dynamics is exactly reversible: for any
macroscopic :math:`N` the *expected* Poincare recurrence time is
astronomically long, while relaxation to (statistical) equilibrium is
essentially immediate.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys.chapters.ehrenfest_urn import EhrenfestUrn
from physicskit.statphys.visualizers.urn_render import plot_ehrenfest_history

# %%
# Relaxation from a maximally ordered initial state
# ------------------------------------------------------
urn = EhrenfestUrn(n_balls=500, seed=0)
history = urn.run(n_steps=5000)

plot_ehrenfest_history(history, n_balls=urn.n_balls)
plt.tight_layout()

print(f"Entropy: S(0)={history['entropy'][0]:.2f}, S(final)={history['entropy'][-1]:.2f}")
print(f"Maximum possible entropy (N/2 balls per box): {EhrenfestUrn(n_balls=500, n_left_init=250).entropy():.2f}")

# %%
# A finite-size collapse: relaxation is fast for any N
# ------------------------------------------------------
# The single N=500 trace above already looks fast, but "fast compared to
# what?" only has an answer once N is varied. Rescaling the occupancy by N
# and time by N (the natural unit: on average N/2 of the N balls need to
# switch box) collapses every system size onto the same relaxation shape,
# while the *fluctuation* band around equilibrium visibly narrows as N
# grows -- the same law of large numbers that makes a macroscopic urn's
# approach to equilibrium look deterministic even though the underlying
# per-ball dynamics is exactly reversible.
N_values = [50, 200, 800, 3200]
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
for N in N_values:
    scan_urn = EhrenfestUrn(n_balls=N, seed=0)
    n_steps = 8 * N
    scan_history = scan_urn.run(n_steps=n_steps)
    t_rescaled = scan_history["t"] / N
    occupancy_rescaled = scan_history["n_left"] / N
    axes[0].plot(t_rescaled, occupancy_rescaled, linewidth=0.8, label=f"N={N}")
    axes[1].plot(t_rescaled, scan_history["entropy"] / N, linewidth=0.8, label=f"N={N}")
axes[0].axhline(0.5, color="k", linestyle="--", linewidth=1, alpha=0.6)
axes[0].set_xlabel("t / N")
axes[0].set_ylabel("(balls in left box) / N")
axes[0].set_title("Relaxation collapses onto one curve in t/N")
axes[0].legend(fontsize=8)
axes[1].set_xlabel("t / N")
axes[1].set_ylabel("entropy per ball, $S/(N k_B)$")
axes[1].set_title("Larger N: relatively smaller entropy fluctuations")
plt.tight_layout()
plt.show()

print(
    "\nRescaled relaxation is essentially N-independent, while the "
    f"root-mean-square late-time fluctuation in n_left/N shrinks with N "
    f"(std over last 20% of run, N={N_values[-1]}): "
    f"{np.std(scan_history['n_left'][int(0.8 * n_steps):] / N):.4f}"
)
