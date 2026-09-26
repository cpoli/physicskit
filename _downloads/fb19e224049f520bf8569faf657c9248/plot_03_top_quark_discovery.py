r"""
CDF and D0: discovery of the top quark
=========================================

The top quark, found at Fermilab's Tevatron in 1995, is so heavy (about
173 GeV) that it decays before it can form a hadron, almost always as
:math:`t\to bW`. In the "lepton + jets" channel one top's :math:`W`
decays to a lepton and neutrino and the other's to two quark jets, so
one top is fully visible as three jets: a :math:`b` jet plus a jet pair
whose mass is :math:`M_W`.

The hard part is combinatorics. An event has four or more jets and no
label saying which came from which top. This example generates
:math:`t\to bW\to bq\bar q'` decays with chained
:func:`~physicskit.particle.decays.two_body_decay` and
:func:`~physicskit.particle.kinematics.boost_generic`, adds the other
top's :math:`b` jet, smears jet energies by 10%, and reconstructs
the three-jet mass with
:func:`~physicskit.particle.kinematics.invariant_mass`. It uses the
:math:`W`-mass constraint and a :math:`b` tag to choose between
assignments, and compares with a W+jets background that has no top at
all.
"""

# %%
import matplotlib.pyplot as plt
import numpy as np

from physicskit.particle.decays import two_body_decay
from physicskit.particle.kinematics import FourVector, boost_generic, invariant_mass

M_t, M_W, m_b = 172.5, 80.4, 4.8
rng = np.random.default_rng(1995)


def iso():
    return rng.uniform(-1, 1), rng.uniform(0, 2 * np.pi)


def jet(q, resolution=0.10):
    """Massless jet with a smeared energy."""
    E = q.E * (1 + resolution * rng.standard_normal())
    d = q.p_vec / q.p_mag
    return FourVector(E, *(E * d))


def boost_random(fv, speed_scale=0.4):
    beta = rng.normal(0, speed_scale, 3)
    beta *= min(1.0, 0.8 / np.linalg.norm(beta))
    return boost_generic(fv, beta), beta


def top_event():
    """Jets from t -> b W(-> q q'), plus the b jet from the other (leptonic) top."""
    b, W = two_body_decay(M_t, m_b, M_W, *iso())
    q1, q2 = two_body_decay(M_W, 0.0, 0.0, *iso())
    beta_W = W.p_vec / W.E
    q1, q2 = boost_generic(q1, beta_W), boost_generic(q2, beta_W)
    parts = []
    _, beta_t = boost_random(b)
    for p in (b, q1, q2):
        parts.append(boost_generic(p, beta_t))
    b_other, _ = two_body_decay(M_t, m_b, M_W, *iso())
    b_other, _ = boost_random(b_other)
    return [jet(p) for p in parts], jet(b_other)


def reconstruct(b_jets, light_jets):
    """Pick the light-jet pair closest to M_W, then the b jet giving the larger three-jet pT."""
    pairs = [(i, j) for i in range(len(light_jets)) for j in range(i + 1, len(light_jets))]
    i, j = min(pairs, key=lambda ij: abs(invariant_mass([light_jets[ij[0]], light_jets[ij[1]]]) - M_W))
    W_cand = light_jets[i] + light_jets[j]
    candidates = [W_cand + b for b in b_jets]
    best = max(candidates, key=lambda c: np.hypot(c.px, c.py))
    return invariant_mass([best]), invariant_mass([light_jets[i], light_jets[j]])


# %%
# Signal: top pairs
# ---------------------
# Both :math:`b` jets are tagged, but the tag doesn't say which top each
# came from, so the reconstruction must choose.
m3_sig, mjj_sig = [], []
for _ in range(4000):
    (b, q1, q2), b_other = top_event()
    m3, mjj = reconstruct([b, b_other], [q1, q2])
    m3_sig.append(m3)
    mjj_sig.append(mjj)
m3_sig = np.array(m3_sig)
print(f"signal: median three-jet mass {np.median(m3_sig):.1f} GeV (input M_t = {M_t})")
print(f"signal: dijet mass {np.median(mjj_sig):.1f} GeV (the W inside the top)")

# %%
# Background: W + jets with no top
# ------------------------------------
# Jets with a falling energy spectrum and random directions; the same
# reconstruction produces a broad, featureless three-jet mass.
m3_bkg = []
for _ in range(4000):
    jets = []
    for _ in range(4):
        E = 20 + rng.exponential(35)
        d = rng.normal(size=3)
        d /= np.linalg.norm(d)
        jets.append(FourVector(E, *(E * d)))
    m3_bkg.append(reconstruct(jets[:2], jets[2:])[0])
m3_bkg = np.array(m3_bkg)

in_window = lambda m: np.mean(np.abs(m - M_t) < 20)  # noqa: E731
print(f"fraction within 20 GeV of M_t: signal {in_window(m3_sig):.2f}, background {in_window(m3_bkg):.2f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
bins = np.linspace(50, 350, 61)
ax1.hist(m3_bkg, bins=bins, color="0.7", label="W + jets background")
ax1.hist(m3_sig, bins=bins, color="firebrick", alpha=0.7, label=r"$t\bar t$ signal")
ax1.axvline(M_t, color="k", ls="--", label=f"$M_t$ = {M_t} GeV")
ax1.set_xlabel("reconstructed three-jet mass [GeV]")
ax1.set_ylabel("events")
ax1.set_title("Top-quark mass from b + two jets")
ax1.legend(fontsize=8)
ax2.hist(mjj_sig, bins=np.linspace(40, 120, 41), color="steelblue", alpha=0.85)
ax2.axvline(M_W, color="k", ls="--", label=f"$M_W$ = {M_W} GeV")
ax2.set_xlabel("dijet mass [GeV]")
ax2.set_title("The W inside the top")
ax2.legend(fontsize=8)
fig.tight_layout()

plt.show()
