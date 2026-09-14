r"""
The Rattleback: one-way spin reversal
==========================================

A rattleback (Celt stone) spins smoothly forever in one sense, but spun
the other way it soon wobbles, stalls, and reverses into the stable
sense of spin.
:class:`~physicskit.classical.systems.rotations.Rattleback` is a
deliberately simplified toy model built directly on
:class:`~physicskit.classical.systems.rotations.EulerTop`'s
free-rigid-body equations, for a reduced state
:math:`(n_1, n_2, n_3)` (:math:`n_3` playing the role of spin about the
roughly-vertical axis, :math:`n_1, n_2` the two rocking/tipping-mode
rates):

.. math::

    \dot n_1 = \frac{I_2 - I_3}{I_1} n_2 n_3 + \gamma n_3^2
        - \mu n_1 - \eta n_1^3, \\
    \dot n_2 = \frac{I_3 - I_1}{I_2} n_3 n_1
        - \mu n_2 - \eta n_2^3, \\
    \dot n_3 = \frac{I_1 - I_2}{I_3} n_1 n_2 - \mu n_3 ,

a damped Euler top plus a single term (:math:`\gamma n_3^2`, added to
the :math:`n_1` equation only) that breaks the :math:`n_3 \to -n_3`
symmetry an ordinary (non-reversing) Euler top has, standing in for the
real rolling-contact-point physics, plus linear damping :math:`\mu` and
cubic self-saturation :math:`\eta` so the instability it pumps up
saturates and decays rather than diverging.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.classical.systems.rotations import Rattleback
from physicskit.classical.visualizers.animations import animate_rattleback

# %%
# A hard spin started predominantly about n3
# ------------------------------------------------
# With gamma > 0, spin started mostly in n3 (with only a tiny rocking
# seed in n1, n2) collapses, overshoots into reversed spin, and decays.

system = Rattleback(n0=[0.01, 0.01, 3.0], I1=1.0, I2=1.5, I3=2.0, gamma=1.0, mu=0.01, eta=0.01)
result = system.integrate((0.0, 400.0), dt=0.1, method="rk4")

# %%
# Animation: spin direction and n3(t) side by side
# --------------------------------------------------------
# The reversal is directly visible two ways: the indicator arrow
# reversing its sense of rotation, and the n3(t) trace crossing zero.
anim = animate_rattleback(system, result, interval=30, stride=20)

plt.show()

# %%
# To save the animation to a file instead of (or in addition to) displaying
# it interactively, use e.g.::
#
#     anim.save("rattleback_animation.gif", writer="pillow", fps=30)

# %%
# With gamma = 0: no reversal
# ---------------------------------
# Setting gamma to zero removes the symmetry-breaking term entirely,
# reducing the system to a damped Euler top that just spins down --
# no reversal, regardless of which way it is spun.
system_no_reversal = Rattleback(n0=[0.01, 0.01, 3.0], I1=1.0, I2=1.5, I3=2.0, gamma=0.0, mu=0.01, eta=0.01)
result_no_reversal = system_no_reversal.integrate((0.0, 400.0), dt=0.02, method="rk4")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(result.t, result.y[:, 2], label="gamma = 1.0 (reverses)", color="steelblue")
ax.plot(result_no_reversal.t, result_no_reversal.y[:, 2], label="gamma = 0.0 (no reversal)", color="gray")
ax.axhline(0.0, color="k", ls="--", lw=0.7)
ax.set_xlabel("t")
ax.set_ylabel(r"$n_3$ (spin rate)")
ax.set_title("Rattleback: spin reversal requires the symmetry-breaking term")
ax.legend()
fig.tight_layout()

plt.show()

print(f"sign changes with gamma=1.0: {np.sum(np.diff(np.sign(result.y[:, 2])) != 0)}")
print(f"sign changes with gamma=0.0: {np.sum(np.diff(np.sign(result_no_reversal.y[:, 2])) != 0)}")
