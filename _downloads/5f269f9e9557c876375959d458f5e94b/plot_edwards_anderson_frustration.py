r"""
Spin glasses: frustration and the Edwards-Anderson order parameter
=========================================================================

A ferromagnet's bonds all agree on which alignment lowers energy. A spin
glass's bonds are quenched random variables -- some ferromagnetic, some
antiferromagnetic -- so a generic plaquette is *frustrated*: no spin
configuration can satisfy all four of its bonds simultaneously. The
Edwards-Anderson model uses the same Ising Hamiltonian form as a
ferromagnet, but with random bond couplings on a periodic :math:`L \times
L` lattice:

.. math::

    H = -\sum_{\langle i,j \rangle} J_{ij}\, s_i s_j, \qquad
    J_{ij} = \pm J \ \text{with equal probability},

where each nearest-neighbor bond :math:`J_{ij}` is independently fixed
("quenched") to :math:`+J` or :math:`-J` before the simulation starts and
:math:`s_i = \pm 1`. The result is a rugged energy landscape riddled with
local minima rather than one clean ground state, and a phase transition
that conventional magnetization cannot detect.

Edwards and Anderson's 1975 fix was to compare two independently
thermalized replicas, :math:`s^{(1)}` and :math:`s^{(2)}`, evolving under
the *same* quenched disorder :math:`\{J_{ij}\}`. Their overlap

.. math::

    q = \frac{1}{N} \sum_i s_i^{(1)} s_i^{(2)}

fluctuates around a nonzero value below the spin-glass transition, where
both replicas freeze into the same disorder-selected (if disordered)
pattern, even though the ordinary magnetization stays zero throughout.
This example uses the mean-squared overlap :math:`\langle q^2 \rangle` as
the spin-glass order parameter.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.statphys.chapters.spin_glass import EdwardsAndersonSpinGlass2D
from physicskit.statphys.visualizers.lattice_render import plot_spin_grid

# %%
# Frustrated plaquettes in a random-bond sample
# ---------------------------------------------------
# An :math:`L=32` lattice with bond magnitude :math:`J=1.0` (each bond
# independently :math:`+J` or :math:`-J`) is generated; a plaquette is
# frustrated when the product of its four surrounding bonds is negative,
# which happens for about half of all plaquettes regardless of the spin
# configuration.
model = EdwardsAndersonSpinGlass2D(L=32, J=1.0, seed=0)
print(f"Frustrated plaquette fraction: {model.frustration_density():.3f} (expected ~0.5 for +/-J bonds)")

model.sweep(beta=2.0, n_sweeps=500)
fig, ax = plt.subplots(figsize=(5, 5))
plot_spin_grid(model.spins, ax=ax, title="Spin glass configuration at $T=0.5\\,J/k_B$")
plt.tight_layout()

# %%
# The Edwards-Anderson order parameter versus temperature
# --------------------------------------------------------------
# Because 2D +/-J spin glasses have no finite-temperature transition
# (:math:`T_{\text{SG}} = 0`), :math:`\langle q^2 \rangle` should decay
# smoothly toward 0 as :math:`T` increases rather than showing a sharp
# transition -- in contrast to the sharp Ising ferromagnetic transition,
# and a useful point of comparison with the Ising examples.
#
# Glassy dynamics make this measurement expensive: at low temperature,
# Metropolis replicas started from random configurations freeze into
# different metastable valleys long before they equilibrate, and
# :math:`\langle q^2 \rangle` then reads near zero for the wrong reason.
# Small lattices, long equilibration, temperatures down to :math:`T = 0.6`
# only, and an average over 16 bond realizations (the quenched average
# :math:`[\langle q^2 \rangle]`) keep the estimate honest.
temperatures = np.linspace(0.6, 3.0, 9)


def disorder_averaged_q2(L, n_disorder=16):
    return np.mean(
        [
            [EdwardsAndersonSpinGlass2D(L=L, J=1.0, seed=s).edwards_anderson_order_parameter(beta=1.0 / T, n_equil=5000, n_measure=1000) for T in temperatures]
            for s in range(n_disorder)
        ],
        axis=0,
    )


q2_by_L = {8: disorder_averaged_q2(8)}

plt.figure(figsize=(6, 4))
plt.plot(temperatures, q2_by_L[8], marker="o")
plt.xlabel("Temperature")
plt.ylabel(r"$[\langle q^2 \rangle]$")
plt.title("Edwards-Anderson replica overlap, L = 8")
plt.tight_layout()

# %%
# No sharp finite-size onset: repeating the sweep at several L
# ------------------------------------------------------------------
# A genuine finite-temperature transition would sharpen into a more and
# more abrupt onset as L grows, the way the Ising ferromagnetic transition
# does elsewhere in this gallery. The 2D +/-J Edwards-Anderson glass is
# believed to order only at :math:`T_{\text{SG}}=0`, so repeating the same
# overlap measurement at several lattice sizes should instead show curves
# that decay smoothly at every L and drop as L grows (the spin-glass
# correlation length stays finite at every T > 0), with no growing
# sharpness -- the qualitative signature that distinguishes a T=0
# transition from a conventional finite-T one.
for L in (12, 16):
    q2_by_L[L] = disorder_averaged_q2(L)
plt.figure(figsize=(6.5, 4.5))
for L, q2_L in q2_by_L.items():
    plt.plot(temperatures, q2_L, marker="o", ms=4, label=f"L={L}")
plt.xlabel("Temperature")
plt.ylabel(r"$[\langle q^2 \rangle]$")
plt.title("Edwards-Anderson overlap: no sharpening onset with L\n(consistent with $T_{SG}=0$)")
plt.legend()
plt.tight_layout()
plt.show()

# %%
# Check
# -----
# Random +-J bonds frustrate half the plaquettes. The overlap is large at
# low T and decays smoothly with T at every size; at fixed T it falls as L
# grows, as it must without a finite-temperature transition.
assert abs(model.frustration_density() - 0.5) < 0.05
for q2_L in q2_by_L.values():
    assert q2_L[0] > 0.1 and q2_L[-1] < 0.05
    assert np.all(np.diff(q2_L) < 0.02)
assert q2_by_L[16][0] < q2_by_L[12][0] < q2_by_L[8][0]
