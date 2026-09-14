r"""
Mandelstam variables and relativistic collision kinematics
=================================================================

Mandelstam (1958) introduced three Lorentz-invariant combinations of a
2-to-2 collision's four-momenta, :math:`s=(p_1+p_2)^2`,
:math:`t=(p_1-p_3)^2`, :math:`u=(p_1-p_4)^2`, satisfying
:math:`s+t+u=\sum_i m_i^2` for on-shell particles -- frame-independent
coordinates for any scattering amplitude or cross section. This example
computes all three with :func:`~physicskit.particle.scattering.mandelstam_s`,
:func:`~physicskit.particle.scattering.mandelstam_t`, and
:func:`~physicskit.particle.scattering.mandelstam_u` for an elastic
:math:`2\to2` scatter, checks the sum rule, and checks that :math:`s`
itself -- unlike the lab-frame energy of either particle -- comes out
identical whether evaluated in the lab frame or after boosting the whole
event into a new frame with
:func:`~physicskit.particle.kinematics.boost_to_com`.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.kinematics import FourVector, boost_generic, boost_to_com
from physicskit.particle.scattering import mandelstam_s, mandelstam_t, mandelstam_u

# %%
# An elastic 2-to-2 collision: equal masses, an arbitrary scattering angle
# --------------------------------------------------------------------------------
m = 1.0
E_beam = 3.0
p_beam = np.sqrt(E_beam**2 - m**2)

p1 = FourVector(E_beam, 0.0, 0.0, p_beam)  # incoming beam particle
p2 = FourVector(m, 0.0, 0.0, 0.0)  # target at rest

# Elastic scattering at angle theta in the CM frame is easiest to build
# there; go to the CM frame, scatter elastically, then boost the whole
# final state back to the lab.
beta_com = boost_to_com([p1, p2])
p1_com = boost_generic(p1, -beta_com)
p2_com = boost_generic(p2, -beta_com)
p_cm = p1_com.p_mag  # |p| is common to both beams in the CM frame for equal masses... here just use p1's

theta = np.radians(40.0)
p3_com = FourVector(p1_com.E, p_cm * np.sin(theta), 0.0, p_cm * np.cos(theta))
p4_com = FourVector(p2_com.E, -p_cm * np.sin(theta), 0.0, -p_cm * np.cos(theta))

p3 = boost_generic(p3_com, beta_com)
p4 = boost_generic(p4_com, beta_com)

# %%
# The three invariants, and the sum rule
# --------------------------------------------
s = mandelstam_s(p1, p2)
t = mandelstam_t(p1, p3)
u = mandelstam_u(p1, p4)
mass_sum = p1.mass**2 + p2.mass**2 + p3.mass**2 + p4.mass**2

print(f"s = {s:.6f}")
print(f"t = {t:.6f}")
print(f"u = {u:.6f}")
print(f"s + t + u = {s + t + u:.6f}")
print(f"sum of m_i^2 = {mass_sum:.6f}  (should match s+t+u exactly, for on-shell particles)")

# %%
# s is frame-independent; the lab-frame energy of a single particle is not
# ------------------------------------------------------------------------------------
# Boost the *entire* event into an arbitrary new frame and recompute
# everything: t and u involve specific particles' momenta and depend on
# the frame in exactly the way any four-vector's components do, but s,
# t, and u are each individually still Lorentz *invariant* -- their
# numerical values, unlike E or p alone, do not change at all.
beta_new_frame = np.array([0.0, 0.0, 0.4])
p1_boosted = boost_generic(p1, beta_new_frame)
p2_boosted = boost_generic(p2, beta_new_frame)
p3_boosted = boost_generic(p3, beta_new_frame)
p4_boosted = boost_generic(p4, beta_new_frame)

s_boosted = mandelstam_s(p1_boosted, p2_boosted)
t_boosted = mandelstam_t(p1_boosted, p3_boosted)
u_boosted = mandelstam_u(p1_boosted, p4_boosted)

print(f"\nafter boosting the whole event by beta={beta_new_frame[2]}:")
print(f"  particle 1's lab-frame energy: {p1.E:.6f}  ->  {p1_boosted.E:.6f}  (changes -- not invariant)")
print(f"  s: {s:.6f}  ->  {s_boosted:.6f}  (unchanged to machine precision -- s IS invariant)")
print(f"  t: {t:.6f}  ->  {t_boosted:.6f}  (unchanged)")
print(f"  u: {u:.6f}  ->  {u_boosted:.6f}  (unchanged)")

# %%
# t and u vs. scattering angle, at fixed s
# ----------------------------------------------
# Sweeping the CM scattering angle at fixed beam energy traces out how
# the momentum-transfer invariants move while s (fixed by the beam
# energy alone) stays put -- t=0 at theta=0 (forward, no momentum
# transferred to particle 3) and u=0 at theta=180 (backward).
theta_values = np.linspace(0.01, np.pi - 0.01, 200)
t_values, u_values = [], []
for th in theta_values:
    p3c = FourVector(p1_com.E, p_cm * np.sin(th), 0.0, p_cm * np.cos(th))
    p4c = FourVector(p2_com.E, -p_cm * np.sin(th), 0.0, -p_cm * np.cos(th))
    t_values.append(mandelstam_t(p1_com, p3c))
    u_values.append(mandelstam_u(p1_com, p4c))

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(np.degrees(theta_values), t_values, color="steelblue", label="t")
ax.plot(np.degrees(theta_values), u_values, color="firebrick", label="u")
ax.axhline(0, color="0.7", lw=0.8)
ax.set_xlabel(r"CM scattering angle $\theta$ (degrees)")
ax.set_ylabel("Mandelstam invariant")
ax.set_title(f"t and u vs. scattering angle, at fixed s={s:.3f}")
ax.legend()
fig.tight_layout()

plt.show()
