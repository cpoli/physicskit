r"""
Cowan and Reines: direct detection of the neutrino
========================================================

More than two decades after Pauli's postulate, Cowan and Reines caught
the "undetectable" particle directly, searching for inverse beta decay,

.. math::

    \bar\nu_e + p \;\longrightarrow\; n + e^{+},

next to the intense antineutrino flux of a nuclear reactor. This
example treats the reaction as the same kind of two-body-like fixed
final-state-momentum kinematics used throughout this chronology's weak
decays -- :func:`~physicskit.particle.decays.two_body_decay_momentum`
gives the fixed final-state momentum once the available energy is
known -- and checks the reaction's energy threshold, the minimum
antineutrino energy for which it can occur at all against a proton at
rest.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.decays import two_body_decay_momentum

# %%
# The energy threshold
# --------------------------
# In the lab frame with the proton at rest, the reaction is
# energetically allowed only once the antineutrino carries enough
# energy to produce the (heavier) neutron and the positron at rest
# in the final center-of-mass frame -- the standard threshold
# calculation for a 2-to-2 reaction with one particle initially at rest.
m_p = 938.272
m_n = 939.565
m_e = 0.511

# Threshold: s_min = (m_n + m_e)^2, and s = m_p^2 + 2*m_p*E_nu (massless
# nu_bar hitting a proton at rest) -> solve for E_nu.
s_min = (m_n + m_e) ** 2
E_nu_threshold = (s_min - m_p**2) / (2.0 * m_p)
print("reaction: nu_bar_e + p -> n + e+")
print(f"threshold antineutrino energy: {E_nu_threshold:.4f} MeV")
print("(a reactor's antineutrino spectrum, peaking around a few MeV, sits comfortably above this --")
print(" exactly why a reactor was the practical neutrino source Cowan and Reines chose)")

# %%
# Final-state kinematics well above threshold
# --------------------------------------------------
# At a typical reactor antineutrino energy, well above threshold, the
# center-of-mass energy fixes a definite final-state momentum for the
# neutron-positron pair, via the same Kallen-function machinery used for
# every two-body decay elsewhere in this chronology.
E_nu = 4.0  # MeV, a typical reactor antineutrino energy
s = m_p**2 + 2.0 * m_p * E_nu
sqrt_s = np.sqrt(s)
p_star = two_body_decay_momentum(sqrt_s, m_n, m_e)
print(f"\nat a typical reactor antineutrino energy E_nu={E_nu} MeV:")
print(f"  center-of-mass energy sqrt(s) = {sqrt_s:.4f} MeV")
print(f"  fixed final-state momentum p* = {p_star:.4f} MeV/c")
print("  (the delayed-coincidence signature Cowan and Reines actually detected: a prompt e+ annihilation")
print("   flash, set by this momentum and the positron's kinetic energy, followed by a delayed neutron-capture flash)")

# %%
# How far below threshold looks different
# ---------------------------------------------
E_nu_values = np.linspace(0.5, 10.0, 100)
s_values = m_p**2 + 2.0 * m_p * E_nu_values
allowed = s_values >= s_min
p_star_values = np.full_like(E_nu_values, np.nan)
p_star_values[allowed] = np.array([two_body_decay_momentum(np.sqrt(s), m_n, m_e) for s in s_values[allowed]])

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(E_nu_values, p_star_values, color="steelblue")
ax.axvline(E_nu_threshold, color="firebrick", ls="--", label=f"threshold = {E_nu_threshold:.3f} MeV")
ax.set_xlabel(r"antineutrino energy $E_{\bar\nu_e}$ (MeV)")
ax.set_ylabel("final-state momentum p* (MeV/c)")
ax.set_title("Inverse beta decay: forbidden below threshold, then rising")
ax.legend()
fig.tight_layout()

plt.show()
