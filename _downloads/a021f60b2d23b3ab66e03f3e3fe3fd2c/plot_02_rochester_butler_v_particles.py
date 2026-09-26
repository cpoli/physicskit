r"""
Rochester and Butler: V particles and the first kaons
========================================================

In 1947 Rochester and Butler found two cloud-chamber photographs among
thousands of cosmic-ray pictures showing an inverted "V": two charged
tracks springing from a point in empty gas, well downstream of the lead
plate where the cosmic ray had interacted. A neutral particle had been
made in the plate, flown a few centimetres unseen, and decayed into two
charged ones. From the tracks' momenta and opening angle they put its
mass at roughly 800 electron masses (modern :math:`K^0`: 497.6 MeV, or
974 electron masses).

This example generates such "V0" decays, :math:`K^0_S\to\pi^+\pi^-`,
with :func:`~physicskit.particle.decays.two_body_decay` and
:func:`~physicskit.particle.kinematics.boost_generic`, draws them in a
magnetic field with
:func:`~physicskit.particle.collider.charged_track_points`, and
reconstructs the parent mass with
:func:`~physicskit.particle.kinematics.invariant_mass` from smeared
track measurements. It also shows the puzzle that earned these particles
the name "strange": made quickly by the strong force, they decay 10
trillion times more slowly.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.collider import charged_track_points
from physicskit.particle.decays import two_body_decay
from physicskit.particle.kinematics import FourVector, boost_generic, invariant_mass

m_K, m_pi = 497.611, 139.570  # MeV
c_tau = 2.684  # cm, K0_S
rng = np.random.default_rng(1947)


def v0_event(p_K, direction):
    """One K0_S -> pi+ pi- decay in the lab: returns (decay point [cm], pi+, pi-)."""
    ct, ph = rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)
    pi_plus, pi_minus = two_body_decay(m_K, m_pi, m_pi, ct, ph)
    beta = p_K / np.hypot(p_K, m_K) * direction
    gamma_beta = p_K / m_K
    L = rng.exponential(gamma_beta * c_tau)
    return L * direction, boost_generic(pi_plus, beta), boost_generic(pi_minus, beta)


# %%
# What the chamber sees
# -------------------------
# Kaons of 0.5-2 GeV/c leave a lead plate at the top (y = 0) heading
# downward, fly a few centimetres, and decay. Only the pion tracks are
# visible, bending oppositely in a 1.5 T field (r [cm] = p [MeV] / 3B);
# they are drawn projected onto the plane of the page.
fig1, ax1 = plt.subplots(figsize=(6, 6))
ax1.axhspan(0, 1.0, color="0.6", label="lead plate")
for _ in range(6):
    direction = np.array([rng.uniform(-0.25, 0.25), -1.0, 0.0])
    direction /= np.linalg.norm(direction)
    vertex, pp, pm = v0_event(rng.uniform(500, 2000), direction)
    ax1.plot([0, vertex[0]], [0, vertex[1]], ":", color="0.6", lw=0.8)
    for pion, q, color in [(pp, +1, "firebrick"), (pm, -1, "steelblue")]:
        planar = FourVector(pion.E, pion.px, pion.py, 0.0)
        pts = charged_track_points(planar, charge=q, B=3.0 * 1.5, vertex=vertex[:2], path_length=25.0)
        ax1.plot(pts[:, 0], pts[:, 1], color=color, lw=1.2)
    ax1.plot(*vertex[:2], "k.", ms=6)
ax1.set_xlim(-20, 20)
ax1.set_ylim(-35, 2)
ax1.set_aspect("equal")
ax1.set_xlabel("x [cm]")
ax1.set_ylabel("y [cm]")
ax1.set_title(r"V0 events: $K^0_S\to\pi^+\pi^-$ (dotted: unseen kaon)")
fig1.tight_layout()

# %%
# Reconstructing the mass of the invisible parent
# ---------------------------------------------------
# For each V, measure both pion momenta (curvature, 4% resolution) and
# directions (1 degree), assume pion masses, and compute the pair's
# invariant mass. The unseen neutral's mass appears as a peak.
masses, decay_lengths, momenta = [], [], []
for _ in range(3000):
    direction = rng.normal(size=3)
    direction /= np.linalg.norm(direction)
    p_K = rng.uniform(500, 2000)
    vertex, pp, pm = v0_event(p_K, direction)
    measured = []
    for pion in (pp, pm):
        p = pion.p_vec * (1 + 0.04 * rng.standard_normal())
        p = p + np.linalg.norm(p) * np.radians(1.0) * rng.standard_normal(3)
        measured.append(FourVector(np.sqrt(p @ p + m_pi**2), *p))
    masses.append(invariant_mass(measured))
    decay_lengths.append(np.linalg.norm(vertex))
    momenta.append(p_K)
masses, decay_lengths, momenta = map(np.array, (masses, decay_lengths, momenta))
print(f"reconstructed V0 mass: median {np.median(masses):.1f} MeV  ({np.median(masses) / 0.511:.0f} electron masses)")

# %%
# Lifetime and the strangeness puzzle
# ---------------------------------------
# Dividing each decay length by :math:`\beta\gamma = p/m` gives the proper
# decay length, exponential with :math:`c\tau\approx2.7` cm, i.e.
# :math:`\tau\approx9\times10^{-11}` s. A particle made by the strong
# interaction should decay in about the time light takes to cross a
# nucleus, :math:`\sim10^{-23}` s.
proper = decay_lengths / (momenta / m_K)
c_tau_fit = proper.mean()
tau = c_tau_fit / 2.998e10
print(f"fitted proper decay length c*tau = {c_tau_fit:.2f} cm  ->  tau = {tau:.2e} s")
print(f"tau / (strong-interaction time 1e-23 s) = {tau / 1e-23:.0e}")

fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(11, 4))
ax2.hist(masses, bins=60, range=(300, 700), color="darkorchid", alpha=0.8)
ax2.axvline(m_K, color="k", ls="--", label=f"$K^0$ mass, {m_K:.1f} MeV")
ax2.set_xlabel(r"$m(\pi^+\pi^-)$ [MeV]")
ax2.set_ylabel("V0 events")
ax2.set_title("Invariant mass of the two tracks")
ax2.legend(fontsize=8)
ax3.hist(proper, bins=40, range=(0, 15), color="steelblue", alpha=0.8, density=True, label="simulated V0s")
x = np.linspace(0, 15, 200)
ax3.plot(x, np.exp(-x / c_tau_fit) / c_tau_fit, color="orange", label=rf"$e^{{-x/c\tau}}$, $c\tau$ = {c_tau_fit:.2f} cm")
ax3.set_xlabel("proper decay length [cm]")
ax3.set_title("A 'long' lifetime of 0.09 ns")
ax3.legend(fontsize=8)
fig2.tight_layout()

plt.show()
