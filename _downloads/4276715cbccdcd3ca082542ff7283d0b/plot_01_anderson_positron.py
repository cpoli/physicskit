r"""
Anderson's positron: a light, positive track in a cloud chamber
==================================================================

In 1932 Carl Anderson photographed a cosmic-ray track crossing a 6 mm
lead plate in a cloud chamber inside a 1.5 T magnetic field. A charged
track bends into an arc of radius

.. math::

    r\,[\mathrm m] = \frac{p\,[\mathrm{GeV}/c]}{0.3\,B\,[\mathrm T]},

so the curvature measures momentum. The track was noticeably more curved
above the plate (23 MeV/c) than below it (63 MeV/c). Particles lose
energy crossing lead, so it must have entered from below. Travelling
upward, its bending direction said its charge was *positive*. A proton
with 23 MeV/c would have stopped within a few millimetres of gas,
but the track ran on for more than 5 cm. It was positive and as light as
an electron: the positron Dirac had predicted.

This example redraws the track with
:func:`~physicskit.particle.collider.charged_track_points` and repeats
the two deductions: direction of travel from energy loss, and mass from
range.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.collider import charged_track_points
from physicskit.particle.kinematics import FourVector

B = 1.5  # tesla
B_eff = 0.3 * B  # so that r [m] = pT [GeV] / B_eff
m_e = 0.000511
p_below, p_above = 0.063, 0.023  # GeV/c

print(f"radius below the plate: {p_below / B_eff * 100:.1f} cm")
print(f"radius above the plate: {p_above / B_eff * 100:.1f} cm")

# %%
# Redraw the track
# --------------------
# Start 8 cm below the plate moving upward, bend up to the plate, cross
# the plate, and continue with the reduced momentum.
plate_y = 0.0


def track(p, direction, vertex, length):
    px, py = p * direction
    fv = FourVector(np.hypot(p, m_e), px, py, 0.0)
    return charged_track_points(fv, charge=+1.0, B=B_eff, vertex=vertex, n_points=200, path_length=length)


lower = track(p_below, np.array([-0.3, 1.0]) / np.hypot(0.3, 1.0), (0.0, -0.08), 0.12)
lower = lower[lower[:, 1] <= plate_y]
d = lower[-1] - lower[-2]
upper = track(p_above, d / np.linalg.norm(d), lower[-1] + d / np.linalg.norm(d) * 0.006, 0.07)

fig, ax = plt.subplots(figsize=(5.5, 6))
ax.axhspan(-0.003, 0.003, color="0.5", label="6 mm lead plate")
ax.plot(lower[:, 0] * 100, lower[:, 1] * 100, color="steelblue", lw=2, label=f"below: p = {p_below * 1e3:.0f} MeV/c")
ax.plot(upper[:, 0] * 100, upper[:, 1] * 100, color="firebrick", lw=2, label=f"above: p = {p_above * 1e3:.0f} MeV/c")
ax.annotate("", xy=lower[60] * 100, xytext=lower[40] * 100, arrowprops=dict(arrowstyle="->", color="steelblue", lw=2))
ax.set_xlabel("x [cm]")
ax.set_ylabel("y [cm]")
ax.set_aspect("equal")
ax.set_title("Anderson's track (B out of page)")
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()

# %%
# Deduction 1: direction and sign of charge
# ---------------------------------------------
# Momentum fell from 63 to 23 MeV/c across the plate, so the particle went
# from the low-curvature side to the high-curvature side: upward. For an
# upward-moving particle in this field, a clockwise bend means positive
# charge. The same picture read the other way (a particle going down)
# would need it to *gain* 40 MeV/c in the lead.
print(f"\nmomentum lost in the plate: {(p_below - p_above) * 1e3:.0f} MeV/c  -> it travelled upward -> charge +e")

# %%
# Deduction 2: too light to be a proton
# -----------------------------------------
# At 23 MeV/c a proton has only :math:`p^2/2m\approx0.28` MeV of kinetic
# energy and a range in the chamber gas of a few millimetres (roughly
# :math:`2.3\,\mathrm{cm}\times(T/\mathrm{MeV})^{1.8}` for protons in
# air). An electron-mass particle at 23 MeV/c is relativistic and ionizes
# at the minimum rate, crossing the whole chamber.
m_p = 0.938272
T_proton = np.hypot(p_above, m_p) - m_p
range_proton_cm = 2.3 * (T_proton * 1e3) ** 1.8
T_light = np.hypot(p_above, m_e) - m_e
print(f"as a proton:   T = {T_proton * 1e3:.2f} MeV, range about {range_proton_cm * 10:.0f} mm  (track observed > 50 mm)")
print(f"as an e-mass:  T = {T_light * 1e3:.1f} MeV, beta = {p_above / np.hypot(p_above, m_e):.4f} -- minimum ionizing, long track")

plt.show()
