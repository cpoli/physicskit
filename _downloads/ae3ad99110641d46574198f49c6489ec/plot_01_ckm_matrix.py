r"""
Kobayashi and Maskawa: the CKM matrix and three quark generations
========================================================================

Cronin and Fitch's CP violation had no accepted explanation until
Kobayashi and Maskawa (1973) showed that a *third* generation of quarks
-- at the time, an unconfirmed extrapolation -- forces the resulting
three-generation quark-mixing matrix to admit exactly one physical
complex phase, making CP violation an unavoidable feature of the weak
interaction rather than an ad hoc addition. No dedicated CKM function
exists in :mod:`physicskit.particle` (the package's other CP-violation
model, the neutral-kaon system, is phenomenological rather than
built from an underlying mixing matrix); this example builds the
standard three-generation unitary mixing matrix directly from Euler-like
mixing angles and one phase -- the same parametrization structure as the
PMNS lepton-mixing matrix -- checks its unitarity, and shows that the
single complex phase is exactly what a **two**-generation (Cabibbo)
matrix cannot admit.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

# %%
# The two-generation Cabibbo matrix: always real, no CP violation possible
# --------------------------------------------------------------------------------
# A single mixing angle gives an orthogonal (real) 2x2 rotation -- no
# room for a complex phase at all, since any phase on a 2x2 unitary
# matrix's entries can be rotated away by redefining the quark fields'
# overall phases.
theta_c = np.radians(13.02)  # the real Cabibbo angle
V_cabibbo = np.array([[np.cos(theta_c), np.sin(theta_c)], [-np.sin(theta_c), np.cos(theta_c)]])
print("Two-generation Cabibbo matrix (always real):")
print(np.round(V_cabibbo, 6))
print(f"is it unitary? max|V^dagger V - I| = {np.max(np.abs(V_cabibbo.T @ V_cabibbo - np.eye(2))):.2e}")

# %%
# The three-generation CKM matrix: one unavoidable complex phase
# ---------------------------------------------------------------------
# The standard parametrization: three mixing angles (theta_12, theta_23,
# theta_13) and one CP-violating phase delta, combined as a product of
# three complex rotations -- with the phase entering only once a third
# generation exists at all.
theta12, theta23, theta13 = np.radians([13.04, 2.38, 0.201])  # close to the measured CKM angles
delta = np.radians(68.8)  # close to the measured CKM CP phase


def ckm_matrix(t12, t23, t13, delta):
    c12, s12 = np.cos(t12), np.sin(t12)
    c23, s23 = np.cos(t23), np.sin(t23)
    c13, s13 = np.cos(t13), np.sin(t13)
    e_idelta = np.exp(1j * delta)
    return np.array(
        [
            [c12 * c13, s12 * c13, s13 * np.conj(e_idelta)],
            [-s12 * c23 - c12 * s23 * s13 * e_idelta, c12 * c23 - s12 * s23 * s13 * e_idelta, s23 * c13],
            [s12 * s23 - c12 * c23 * s13 * e_idelta, -c12 * s23 - s12 * c23 * s13 * e_idelta, c23 * c13],
        ]
    )


V_ckm = ckm_matrix(theta12, theta23, theta13, delta)
print("\nThree-generation CKM matrix, |V_ij| (close to the measured values):")
print(np.round(np.abs(V_ckm), 6))

unitarity_error = np.max(np.abs(V_ckm.conj().T @ V_ckm - np.eye(3)))
print(f"\nis it unitary? max|V^dagger V - I| = {unitarity_error:.2e}")

# %%
# The Jarlskog invariant: a basis-independent measure of the CP phase
# ------------------------------------------------------------------------------
# A single number that vanishes if and only if there is no CP violation
# -- for the real 2x2 Cabibbo matrix it is identically zero; for the
# CKM matrix with delta != 0 it is not.
J = np.imag(V_ckm[0, 0] * V_ckm[1, 1] * np.conj(V_ckm[0, 1]) * np.conj(V_ckm[1, 0]))
print(f"\nJarlskog invariant J = {J:.3e} (nonzero -- CP violation is unavoidable once three generations mix)")

V_ckm_no_phase = ckm_matrix(theta12, theta23, theta13, 0.0)
J_at_delta0 = np.imag(V_ckm_no_phase[0, 0] * V_ckm_no_phase[1, 1] * np.conj(V_ckm_no_phase[0, 1]) * np.conj(V_ckm_no_phase[1, 0]))
print(f"Jarlskog invariant at delta=0 (hypothetically no phase): {J_at_delta0:.3e} (vanishes -- delta is exactly what makes J nonzero)")

# %%
# Visualizing the mixing strengths
# --------------------------------------
fig, ax = plt.subplots(figsize=(5.5, 5))
im = ax.imshow(np.abs(V_ckm), cmap="viridis", vmin=0, vmax=1)
ax.set_xticks([0, 1, 2])
ax.set_xticklabels(["d", "s", "b"])
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(["u", "c", "t"])
for i in range(3):
    for j in range(3):
        ax.text(j, i, f"{np.abs(V_ckm[i, j]):.3f}", ha="center", va="center", color="white" if np.abs(V_ckm[i, j]) < 0.6 else "black")
fig.colorbar(im, ax=ax, label="|V_ij|")
ax.set_title("CKM matrix magnitudes: strongly diagonal, small cross-generation mixing")
fig.tight_layout()

plt.show()
