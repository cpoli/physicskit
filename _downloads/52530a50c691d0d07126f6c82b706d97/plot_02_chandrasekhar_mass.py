r"""
Chandrasekhar's white dwarf mass limit
===========================================

Chandrasekhar (1931) showed that electron degeneracy pressure cannot
support an arbitrarily massive white dwarf: as the star's mass grows,
the degenerate electrons are forced relativistic, softening the
pressure-density relation until, above a critical mass, no equilibrium
exists at all,

.. math::

    M_{\rm Ch} \approx \frac{5.83}{\mu_e^2}\ M_\odot,

the standard coefficient from the :math:`n=3` relativistic-degenerate
polytrope. :func:`~physicskit.astro.stellar_structure.chandrasekhar_mass`
returns exactly this limit; this example plots it against the mean
molecular weight per electron :math:`\mu_e`, and demonstrates the
special property of the :math:`n=3` polytrope behind it directly with
:class:`~physicskit.astro.stellar_structure.PolytropicStar`: unlike any
other polytropic index, its mass does not depend on the star's central
density.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.astro.stellar_structure import PolytropicStar, chandrasekhar_mass

# %%
# The mass limit vs. composition
# ------------------------------------
# :math:`\mu_e=2.0` for a carbon/oxygen white dwarf (equal numbers of
# protons and neutrons, so one electron per two nucleons);
# helium/hydrogen compositions have smaller :math:`\mu_e` and hence a
# (slightly) higher limit.
mu_e_values = np.linspace(1.7, 2.2, 100)
M_ch = np.array([chandrasekhar_mass(mu) for mu in mu_e_values])

fig1, ax1 = plt.subplots(figsize=(6, 4.5))
ax1.plot(mu_e_values, M_ch, color="steelblue")
ax1.axvline(2.0, color="0.6", ls="--", lw=1)
ax1.scatter([2.0], [chandrasekhar_mass(2.0)], color="firebrick", zorder=5, label=f"carbon/oxygen WD ($\\mu_e$=2): {chandrasekhar_mass(2.0):.3f} $M_\\odot$")
ax1.set_xlabel(r"mean molecular weight per electron, $\mu_e$")
ax1.set_ylabel(r"$M_{\rm Ch}$ ($M_\odot$)")
ax1.set_title("Chandrasekhar mass vs. composition")
ax1.legend(fontsize=8)
fig1.tight_layout()

print(f"Chandrasekhar mass, mu_e=2.0 (C/O white dwarf): {chandrasekhar_mass(2.0):.4f} solar masses")

# %%
# Why n=3 is special: mass independent of central density
# -----------------------------------------------------------------
# For any polytropic index except n=3, a star's mass depends on its
# central density; :math:`n=3` is the one index -- corresponding to
# fully relativistic electron degeneracy, :math:`P\propto\rho^{4/3}` --
# for which it does not. Building two n=3 stars with the *same*
# polytropic constant K but very different central densities
# demonstrates this directly.
K = 1.0
star_low = PolytropicStar(n=3.0, K=K, rho_c=1.0)
star_high = PolytropicStar(n=3.0, K=K, rho_c=1000.0)
print(f"\nn=3 polytrope, rho_c=1:    mass = {star_low.mass:.8f}, radius = {star_low.radius:.6f}")
print(f"n=3 polytrope, rho_c=1000: mass = {star_high.mass:.8f}, radius = {star_high.radius:.6f}")
print(f"masses agree to a relative difference of {abs(star_low.mass - star_high.mass) / star_low.mass:.2e}, despite a 1000x density difference")
print("(radius, by contrast, does shrink as rho_c increases -- only the mass is density-independent for n=3)")

# For comparison, an n=1.5 polytrope's mass *does* depend on rho_c.
star15_low = PolytropicStar(n=1.5, K=K, rho_c=1.0)
star15_high = PolytropicStar(n=1.5, K=K, rho_c=1000.0)
print("\nfor contrast, n=1.5 polytrope masses at the same two densities:")
print(f"  rho_c=1:    mass = {star15_low.mass:.6f}")
print(f"  rho_c=1000: mass = {star15_high.mass:.6f}  (these differ substantially, unlike the n=3 case)")

fig2, ax2 = plt.subplots(figsize=(5.5, 4.2))
rho_c_values = np.logspace(0, 3, 20)
masses_n3 = [PolytropicStar(n=3.0, K=K, rho_c=rc).mass for rc in rho_c_values]
masses_n15 = [PolytropicStar(n=1.5, K=K, rho_c=rc).mass for rc in rho_c_values]
ax2.semilogx(rho_c_values, masses_n3, "o-", color="firebrick", label="n=3 (mass ~ constant)")
ax2.semilogx(rho_c_values, masses_n15, "o-", color="steelblue", label="n=1.5 (mass varies)")
ax2.set_xlabel(r"central density $\rho_c$")
ax2.set_ylabel("mass")
ax2.set_title("Only n=3 gives a density-independent mass")
ax2.legend(fontsize=8)
fig2.tight_layout()

plt.show()
