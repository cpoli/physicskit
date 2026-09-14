r"""
Einstein's mass-energy equivalence
========================================

In the last of his 1905 papers, Einstein showed that a body's rest mass
and rest energy are the same quantity, :math:`E_0=mc^2` -- the rest-frame
special case of the fully relativistic relation
:math:`E^2=(pc)^2+(mc^2)^2`.
:class:`~physicskit.particle.kinematics.FourVector` builds exactly this
object, with its ``.mass`` property returning the invariant
:math:`\sqrt{E^2-|\vec p|^2}`. This example checks that the invariant
mass a moving observer computes agrees exactly with the rest energy
measured in the particle's own rest frame -- true at every boost
velocity, however fast -- and that summing four-vectors before taking
the mass, rather than after, is what makes invariant mass differ from
simply adding up individual particle masses.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.kinematics import FourVector, boost

# %%
# The rest-frame mass is boost-invariant
# ---------------------------------------------
# A particle of rest mass 1.0, boosted to a sequence of speeds -- the
# invariant mass recovered from E and p agrees with the rest mass at
# every single one, even as E and p themselves grow without bound as
# beta -> 1.
m0 = 1.0
p_rest = FourVector(m0, 0.0, 0.0, 0.0)
betas = np.linspace(0.0, 0.999, 200)
E_values = np.array([boost(p_rest, b).E for b in betas])
p_values = np.array([boost(p_rest, b).p_mag for b in betas])
mass_recovered = np.array([boost(p_rest, b).mass for b in betas])

print(f"rest mass:                    {m0}")
print(f"invariant mass at beta=0.9:   {boost(p_rest, 0.9).mass:.10f}")
print(f"invariant mass at beta=0.999: {boost(p_rest, 0.999).mass:.10f}")
print(f"max |recovered mass - m0| over the whole sweep: {np.max(np.abs(mass_recovered - m0)):.2e}")

fig1, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
ax1.plot(betas, E_values, label="E (grows without bound)", color="steelblue")
ax1.plot(betas, p_values, label="|p| (grows without bound)", color="firebrick")
ax1.set_xlabel(r"$\beta$")
ax1.set_ylabel("E, |p|")
ax1.set_title(r"$E$ and $|\vec p|$ individually diverge as $\beta\to1$")
ax1.legend(fontsize=8)

ax2.plot(betas, mass_recovered, color="darkorchid")
ax2.axhline(m0, color="0.6", ls="--", lw=1)
ax2.set_xlabel(r"$\beta$")
ax2.set_ylabel("recovered mass")
ax2.set_title(r"but $m=\sqrt{E^2-|\vec p|^2}=m_0$ always")
fig1.tight_layout()

# %%
# Why invariant mass is not just the sum of masses
# ------------------------------------------------------
# Two identical particles of rest mass 1.0, fired at each other with
# equal and opposite momenta: their *individual* masses sum to 2, but
# the *invariant* mass of the combined system -- what a detector actually
# reconstructs from the total four-momentum -- is larger, because the
# system also carries the kinetic energy of the collision.
p1 = boost(FourVector(m0, 0.0, 0.0, 0.0), 0.6, axis="z")
p2 = boost(FourVector(m0, 0.0, 0.0, 0.0), -0.6, axis="z")
system_mass = (p1 + p2).mass
print(f"\ntwo particles, each rest mass {m0}, each boosted to beta=+/-0.6:")
print(f"  sum of individual masses:      {p1.mass + p2.mass:.6f}")
print(f"  invariant mass of the SYSTEM:  {system_mass:.6f}  (larger -- it also carries the collision's kinetic energy)")

plt.show()
