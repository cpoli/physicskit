r"""
Cockcroft and Walton: splitting lithium with accelerated protons
===================================================================

In 1932 Cockcroft and Walton accelerated protons through a few hundred
kilovolts and fired them at lithium. Scintillation screens flashed with
pairs of alpha particles flying apart in opposite directions:

.. math::

    {}^1\mathrm{H} + {}^7\mathrm{Li} \to 2\,{}^4\mathrm{He}.

It was the first nuclear reaction produced by a machine, and each alpha
carried about 8.7 MeV, far more than the proton brought in. The extra
energy is the mass lost, :math:`Q=\Delta m\,c^2`. It worked at such low
voltages because of quantum tunnelling: Gamow had just shown that a
proton can get through the Coulomb barrier without having enough energy
to go over it. This example computes the reaction's :math:`Q` with
:func:`~physicskit.particle.nuclear.q_value`, the alpha energies and
ranges, and the tunnelling probability that set the reaction yield.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.nuclear import q_value

U = 931.494  # MeV per atomic mass unit
m_H1, m_Li7, m_He4 = 1.007825, 7.016003, 4.002602

# %%
# Energy released: :math:`Q=\Delta m\,c^2`
# --------------------------------------------
Q = q_value([m_H1, m_Li7], [m_He4, m_He4]) * U
print(f"mass lost: {(m_H1 + m_Li7 - 2 * m_He4) * 1e3:.3f} milli-u  ->  Q = {Q:.3f} MeV")

T_p = 0.25  # MeV, a typical Cockcroft-Walton proton energy
T_alpha = (Q + T_p) / 2  # centre-of-mass share; the lab boost from a slow proton is small
R_air = 0.318 * T_alpha**1.5  # Geiger's range-energy rule for alphas in air, cm
print(f"each alpha: about {T_alpha:.2f} MeV, range in air about {R_air:.1f} cm (observed: 8.4 cm)")
print(f"energy out / energy in: {2 * T_alpha / T_p:.0f}")

# %%
# Why a few hundred keV was enough
# ------------------------------------
# The Coulomb barrier between a proton and lithium at nuclear contact is
# about 1.2 MeV. A classical proton at 0.25 MeV never reaches the nucleus.
# Quantum mechanically it tunnels through with probability close to the
# Gamow factor :math:`e^{-2\pi\eta}`, with
# :math:`\eta = Z_1Z_2\alpha c/v`. That probability climbs steeply with
# energy, which is why the alpha count rose so fast as Cockcroft and Walton
# turned up the voltage.
alpha_fs, m_p = 1 / 137.036, 938.272
r_contact = 1.2 * (1 + 7 ** (1 / 3))  # fm
V_barrier = 1.44 * 3 / r_contact  # MeV (e^2 / 4 pi eps0 = 1.44 MeV fm)
print(f"\nCoulomb barrier at contact ({r_contact:.1f} fm): {V_barrier:.2f} MeV")

mu = m_p * 7 * m_p / (8 * m_p)  # reduced mass (nucleon masses)
E_cm = np.linspace(0.05, 0.8, 300) * 7 / 8
v = np.sqrt(2 * E_cm / mu)
gamow = np.exp(-2 * np.pi * 3 * alpha_fs / v)
yield_rel = gamow / E_cm  # sigma ~ S / E * exp(-2 pi eta), constant S
E_lab = E_cm * 8 / 7
for E in (0.1, 0.25, 0.5):
    print(f"tunnelling factor at {E:.2f} MeV proton energy: {np.interp(E, E_lab, gamow):.2e}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
r = np.linspace(0.3, 12, 400)
ax1.plot(r, np.where(r > r_contact, 1.44 * 3 / r, np.nan), color="k", label="Coulomb barrier")
ax1.plot([0, r_contact], [-10, -10], color="k")
ax1.plot([r_contact, r_contact], [-10, V_barrier], color="k")
ax1.axhline(T_p * 7 / 8, color="steelblue", ls="--", label=f"proton energy (CM), {T_p} MeV lab")
ax1.set_ylim(-1, 2)
ax1.set_xlabel("separation [fm]")
ax1.set_ylabel("potential energy [MeV]")
ax1.set_title("The proton must tunnel")
ax1.legend(fontsize=8)
ax2.semilogy(E_lab, yield_rel / yield_rel[-1], color="firebrick")
ax2.set_xlabel("proton energy [MeV]")
ax2.set_ylabel("relative reaction yield")
ax2.set_title("Gamow tunnelling sets the yield")
fig.tight_layout()

plt.show()
