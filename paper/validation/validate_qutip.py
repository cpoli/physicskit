"""
Cross-check of physicskit's quantum optics, open systems and spin chains against QuTiP.

Each model is built twice, once with physicskit and once from QuTiP's own
operators (``destroy``, ``sigmap``, ``tensor``, ...), and each package then
computes on its own:

1. Jaynes-Cummings: the spectrum, and the collapse-and-revival of the
   atomic inversion from a coherent field (physicskit diagonalizes once,
   QuTiP integrates the Schrodinger equation with ``sesolve``).
2. Coherent and squeezed states, and the Wigner functions of a Fock state,
   a cat state and a displaced squeezed state.
3. A driven, damped atom-cavity system: the Lindblad evolution
   (physicskit exponentiates the Liouvillian, QuTiP integrates ``mesolve``)
   and the steady state.
4. Transverse-field Ising and XXZ chains: the low-lying spectrum, and the
   half-chain entanglement entropy of the ground state.

Each comparison is asserted; ``RESULTS`` holds the largest deviations for
the paper's validation table.
"""

import numpy as np
import qutip as qt

from physicskit.condensed.spin_chains import entanglement_entropy, lowest_eigenstates, tfim_hamiltonian, xxz_hamiltonian
from physicskit.optics.quantum_optics import JaynesCummingsModel, coherent_state, compute_wigner_function, fock_state, squeezed_state
from physicskit.quantum.chapters.open_systems import lindblad_steady_state, solve_lindblad

RESULTS = {}
TIGHT = {"method": "vern9", "atol": 1e-13, "rtol": 1e-12, "nsteps": 10**6}  # QuTiP solver options, well below the checked deviations


def record(name, dev, tol):
    RESULTS[name] = float(dev)
    assert dev < tol, (name, dev)
    print(f"{name:60s} max deviation {dev:.1e}")


# 1. Jaynes-Cummings -------------------------------------------------------

wc, wa, g, cutoff, alpha = 1.0, 1.0, 0.05, 40, np.sqrt(5.0)
jc = JaynesCummingsModel(omega_c=wc, omega_a=wa, g=g, cutoff=cutoff)
a, sm = qt.tensor(qt.destroy(cutoff), qt.qeye(2)), qt.tensor(qt.qeye(cutoff), qt.sigmam())
sz = qt.tensor(qt.qeye(cutoff), qt.sigmaz())
H_qt = wc * a.dag() * a + 0.5 * wa * sz + g * (a * sm.dag() + a.dag() * sm)
record("Jaynes-Cummings: spectrum", np.abs(np.linalg.eigvalsh(jc.hamiltonian()) - H_qt.eigenenergies()).max(), 1e-12)

psi0 = np.kron(coherent_state(alpha, cutoff), [1.0, 0.0])  # coherent field, excited atom
t = np.linspace(0.0, 2.2 * 2 * np.pi * alpha / g, 401)  # past the first revival, t_r = 2 pi |alpha| / g
pe_pk = np.abs(jc.evolve(psi0, t)[:, 0::2]) ** 2
pe_qt = qt.sesolve(H_qt, qt.Qobj(psi0, dims=[[cutoff, 2], [1, 1]]), t, e_ops=[(sz + 1) / 2], options=TIGHT).expect[0]
record("Jaynes-Cummings: inversion, collapse and revival", np.abs(pe_pk.sum(axis=1) - pe_qt).max(), 1e-8)

# 2. States and Wigner functions -------------------------------------------

n = 60
dev_coh = 0.0
for z in (0.8, 1.5 - 0.7j, 3.0j):
    ref = qt.coherent(n, z, method="analytic").full().ravel()
    dev_coh = max(dev_coh, np.abs(coherent_state(z, n) - ref / np.linalg.norm(ref)).max())
dev_sq = 0.0
for xi, al in ((0.5, 0.0), (0.4 * np.exp(0.6j), 1.0 - 0.5j)):
    ref = (qt.displace(n, al) * qt.squeeze(n, xi) * qt.basis(n, 0)).full().ravel()
    dev_sq = max(dev_sq, np.abs(squeezed_state(xi, al, n) - ref / np.linalg.norm(ref)).max())
record("coherent states (Fock amplitudes)", dev_coh, 1e-12)
record("displaced squeezed states (Fock amplitudes)", dev_sq, 1e-12)

x = np.linspace(-5.0, 5.0, 81)
cat = coherent_state(2.0, 30) + coherent_state(-2.0, 30)
dev_w = 0.0
for psi in (fock_state(3, 30), cat / np.linalg.norm(cat), squeezed_state(0.4, 0.5 + 0.5j, 30)):
    W_pk = compute_wigner_function(psi, x, x)  # W[x, p]
    W_qt = qt.wigner(qt.Qobj(psi), x, x)  # W[p, x], same quadratures (g = sqrt 2)
    dev_w = max(dev_w, np.abs(W_pk - W_qt.T).max())
record("Wigner functions: Fock, cat and squeezed states", dev_w, 1e-12)

# 3. Driven, damped atom-cavity system (Lindblad) --------------------------

nc, delta, drive, kappa, gamma = 8, 0.3, 0.4, 0.5, 0.2
a, sm = qt.tensor(qt.destroy(nc), qt.qeye(2)), qt.tensor(qt.qeye(nc), qt.sigmam())
H_qt = delta * (a.dag() * a + sm.dag() * sm) + 0.6 * (a.dag() * sm + sm.dag() * a) + drive * (a + a.dag())
c_qt = [np.sqrt(kappa) * a, np.sqrt(gamma) * sm]
H, c_ops = H_qt.full(), [c.full() for c in c_qt]
rho0 = np.zeros((2 * nc, 2 * nc), complex)
rho0[1, 1] = 1.0  # empty cavity, atom in its ground state
t = np.linspace(0.0, 20.0, 201)
rho_pk = solve_lindblad(rho0, H, c_ops, t)
rho_qt = qt.mesolve(H_qt, qt.Qobj(rho0, dims=H_qt.dims), t, c_qt, options=TIGHT).states
record("driven atom-cavity: Lindblad evolution", max(np.abs(r - s.full()).max() for r, s in zip(rho_pk, rho_qt)), 1e-9)
record("driven atom-cavity: steady state", np.abs(lindblad_steady_state(H, c_ops) - qt.steadystate(H_qt, c_qt).full()).max(), 1e-10)


# 4. Spin chains -----------------------------------------------------------


def chain(N, terms):
    """Sum over the bonds of a periodic chain of N spins-1/2, from QuTiP single-site operators."""

    def site(op, i):
        ops = [qt.qeye(2)] * N
        ops[i] = op
        return qt.tensor(ops)

    H = 0
    for coeff, op1, op2 in terms:
        for i in range(N):
            H += coeff * (site(op1, i) * site(op2, (i + 1) % N) if op2 is not None else site(op1, i))
    return H


N, k = 12, 6
dev_e, dev_s = 0.0, 0.0
for J, h in ((1.0, 0.5), (1.0, 1.0), (1.0, 1.7)):
    H_pk, _ = tfim_hamiltonian(N, J=J, h=h)
    H_qt = chain(N, [(-J, qt.sigmax(), qt.sigmax()), (-h, qt.sigmaz(), None)])
    E_pk, V_pk = lowest_eigenstates(H_pk, k=k)
    E_qt, V_qt = H_qt.eigenstates(sparse=True, eigvals=k)
    dev_e = max(dev_e, np.abs(E_pk - E_qt).max())
    if E_pk[1] - E_pk[0] > 1e-6:  # unique ground state: compare its entanglement
        dev_s = max(dev_s, abs(entanglement_entropy(V_pk[:, 0], N, N // 2) - qt.entropy_vn(V_qt[0].ptrace(list(range(N // 2))))))
record("transverse-field Ising chain (N = 12): lowest 6 levels", dev_e, 1e-10)
record("transverse-field Ising chain: half-chain entanglement entropy", dev_s, 1e-9)

dev_e = 0.0
S = [0.5 * op for op in (qt.sigmax(), qt.sigmay(), qt.sigmaz())]
for delta, h in ((1.0, 0.0), (0.5, 0.3), (2.0, 0.1)):
    H_pk, _ = xxz_hamiltonian(N, J=1.0, delta=delta, h=h)
    H_qt = chain(N, [(1.0, S[0], S[0]), (1.0, S[1], S[1]), (delta, S[2], S[2]), (-h, S[2], None)])
    dev_e = max(dev_e, np.abs(lowest_eigenstates(H_pk, k=k)[0] - H_qt.eigenenergies(sparse=True, eigvals=k)).max())
record("XXZ chain (N = 12): lowest 6 levels", dev_e, 1e-10)
