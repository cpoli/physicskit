r"""
Chadwick's discovery of the neutron
======================================

Bombarding beryllium with polonium alphas produced a penetrating neutral
radiation that knocked protons out of paraffin wax at up to about 5.7
MeV. Curie and Joliot took it to be gamma rays. In 1932 Chadwick showed
the numbers didn't work for photons. A photon would need about 55 MeV
to give a proton that much recoil in a Compton collision, and it would
have to be even more energetic to explain the nitrogen recoils he also
measured. The same energy could not account for both. A neutral particle
with about the proton's mass, colliding elastically, accounts for both
recoils with a single energy.

This example redoes both halves of the argument: the Compton
kinematics, built with :class:`~physicskit.particle.kinematics.FourVector`
and checked with :func:`~physicskit.particle.kinematics.invariant_mass`,
and Chadwick's mass measurement from the maximum recoil speeds of
hydrogen and nitrogen nuclei.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import brentq

from physicskit.particle.kinematics import FourVector, invariant_mass

m_p = 938.272  # MeV
m_N14 = 14.003074 * 931.494 - 7 * 0.511

# %%
# The gamma-ray hypothesis fails
# ----------------------------------
# A photon of energy :math:`E_\gamma` hitting a nucleus of mass :math:`M`
# head on gives it at most :math:`T_{\max} = 2E_\gamma^2/(M+2E_\gamma)`.
# Invert for the photon energy each observed recoil requires.


def compton_T_max(E_gamma, M):
    return 2 * E_gamma**2 / (M + 2 * E_gamma)


T_p_obs, T_N_obs = 5.7, 1.2  # MeV, maximum recoils Chadwick measured
E_for_p = brentq(lambda E: compton_T_max(E, m_p) - T_p_obs, 0.1, 1000)
E_for_N = brentq(lambda E: compton_T_max(E, m_N14) - T_N_obs, 0.1, 1000)
print(f"photon energy needed for a {T_p_obs} MeV proton recoil:  {E_for_p:.0f} MeV")
print(f"photon energy needed for a {T_N_obs} MeV nitrogen recoil: {E_for_N:.0f} MeV")
print("-> no single photon energy explains both")

# Check the head-on collision with explicit four-vectors: photon + proton at rest.
k = FourVector(E_for_p, 0, 0, E_for_p)
p_in = FourVector(m_p, 0, 0, 0)
k_out = FourVector(E_for_p - T_p_obs, 0, 0, -(E_for_p - T_p_obs))  # back-scattered photon
p_out = FourVector(m_p + T_p_obs, 0, 0, E_for_p + (E_for_p - T_p_obs))
print(f"four-momentum check: sqrt(s) in {invariant_mass([k, p_in]):.3f}, out {invariant_mass([k_out, p_out]):.3f} MeV; recoil mass {p_out.mass:.3f} MeV")

# %%
# A neutral particle of mass :math:`\approx m_p`
# --------------------------------------------------
# In an elastic head-on collision, a particle of mass :math:`m` and speed
# :math:`v` gives a target of mass :math:`M` a recoil speed
# :math:`2mv/(m+M)`. Chadwick measured maximum recoil speeds of
# :math:`3.3\times10^9` cm/s for hydrogen and :math:`4.7\times10^8` cm/s
# for nitrogen. Their ratio fixes :math:`m` without knowing :math:`v`:
#
# .. math::
#
#     \frac{v_H}{v_N} = \frac{m+14}{m+1} \quad\Rightarrow\quad
#     m = \frac{14v_N - v_H}{v_H - v_N}.
v_H, v_N = 3.3e9, 4.7e8
m_neutron = (14 * v_N - v_H) / (v_H - v_N)
print(f"\nChadwick's neutron mass from recoil speeds: {m_neutron:.2f} proton masses  (modern: 1.0014)")

# %%
# One neutron energy explains every recoil
# --------------------------------------------
# With :math:`m=1`, a neutron of kinetic energy :math:`E_n` gives a
# nucleus of mass number :math:`A` a maximum recoil energy
# :math:`4A/(1+A)^2\,E_n`. A single :math:`E_n\approx5.7` MeV reproduces
# both the proton and nitrogen recoils; the photon hypothesis needs a
# different energy for each target.
A = np.array([1, 4, 7, 12, 14, 16])
E_n = T_p_obs
T_recoil_n = 4 * A / (1 + A) ** 2 * E_n
print(f"\nneutron of {E_n} MeV: nitrogen recoil {4 * 14 / 15**2 * E_n:.2f} MeV (observed about {T_N_obs} MeV)")

E_gamma_needed = [brentq(lambda E: compton_T_max(E, a * 931.494) - t, 0.1, 5000) for a, t in zip(A, T_recoil_n)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
ax1.plot(A, T_recoil_n, "o-", color="steelblue", label=f"neutron, $E_n$ = {E_n} MeV")
ax1.plot([1, 14], [T_p_obs, T_N_obs], "s", color="k", mfc="none", ms=10, label="Chadwick's measured maxima")
ax1.set_xlabel("target mass number A")
ax1.set_ylabel("maximum recoil energy [MeV]")
ax1.set_title("One neutron energy fits every target")
ax1.legend(fontsize=8)
ax2.plot(A, E_gamma_needed, "o-", color="firebrick")
ax2.set_xlabel("target mass number A")
ax2.set_ylabel("photon energy required [MeV]")
ax2.set_title("The gamma-ray hypothesis needs a different energy per target")
fig.tight_layout()

plt.show()
