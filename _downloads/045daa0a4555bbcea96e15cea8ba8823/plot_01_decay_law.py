r"""
Rutherford and Soddy's law of radioactive decay
===================================================

Rutherford and Soddy (1902-1903), working with thorium compounds, showed
that radioactivity is one element transmuting into another, and that the
number of parent atoms falls off as a simple exponential,

.. math::

    N(t) = N_0\,e^{-\lambda t}, \qquad t_{1/2} = \frac{\ln2}{\lambda},

with decay constant :math:`\lambda` characteristic of the parent species
alone. This example builds that law directly from
:func:`~physicskit.particle.decays.decay_constant`,
:func:`~physicskit.particle.decays.radioactive_decay_number`, and
:func:`~physicskit.particle.decays.activity` -- and checks the defining property
of a half-life: the population halves in every successive interval of
length :math:`t_{1/2}`, regardless of how much has already decayed.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.decays import (
    activity,
    decay_constant,
    half_life,
    radioactive_decay_number,
)

# %%
# From half-life to decay constant, and back
# ------------------------------------------------
# A radium-226-like half-life, in days, for concreteness.
t_half = 1600.0 * 365.25  # ~1600 years, in days
lam = decay_constant(t_half)
print(f"half-life:     {t_half:.3f} days")
print(f"decay constant: {lam:.3e} / day")
print(f"round trip:    half_life(decay_constant(t_half)) = {half_life(lam):.3f} days (should equal t_half)")

# %%
# The exponential decay law
# --------------------------------
N0 = 1.0e6
t = np.linspace(0.0, 5.0 * t_half, 500)
N = radioactive_decay_number(N0, lam, t)
A = activity(N, lam)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(t / t_half, N, color="steelblue")
for k in range(1, 5):
    ax1.axvline(k, color="0.75", lw=0.8, ls="--")
ax1.set_xlabel(r"$t / t_{1/2}$")
ax1.set_ylabel("N(t)")
ax1.set_title("Exponential decay: halves every $t_{1/2}$")

ax2.semilogy(t / t_half, N, color="firebrick")
ax2.set_xlabel(r"$t / t_{1/2}$")
ax2.set_ylabel("N(t) (log scale)")
ax2.set_title("A straight line on a log axis -- the signature of exponential decay")
fig.tight_layout()

# %%
# The defining property: halving in *every* interval of length t_1/2
# --------------------------------------------------------------------------
# Not just from t=0 -- from *any* starting point. This is what makes a
# half-life a property of the isotope alone, independent of how much of
# the original sample remains, or when it was prepared.
for start_multiple in [0, 1, 2, 3]:
    t_start = start_multiple * t_half
    N_start = radioactive_decay_number(N0, lam, t_start)
    N_after_one_halflife = radioactive_decay_number(N0, lam, t_start + t_half)
    ratio = N_after_one_halflife / N_start
    print(f"N({start_multiple}*t_1/2) = {N_start:.4e}  ->  N({start_multiple + 1}*t_1/2) = {N_after_one_halflife:.4e}  (ratio = {ratio:.6f})")

print(f"\nactivity (Becquerel's actual observable, the plate-blackening rate) at t=0: {A[0]:.4e} decays/day")
print(f"activity after one half-life:                                            {activity(radioactive_decay_number(N0, lam, t_half), lam):.4e} decays/day")

plt.show()
