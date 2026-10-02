r"""
Shor's period finding and the quantum Fourier transform
=======================================================

Peter Shor (1994) showed that a quantum computer could factor integers in
polynomial time, which no known classical algorithm does. Factoring
:math:`N` reduces to finding the period :math:`r` of :math:`f(x) = a^x \bmod
N`; then :math:`\gcd(a^{r/2} \pm 1, N)` are factors. The quantum step is the
Fourier transform over :math:`2^t` amplitudes,

.. math::

    |x\rangle \to \frac{1}{\sqrt{2^t}}\sum_{y=0}^{2^t-1} e^{2\pi i xy/2^t}|y\rangle,

which Coppersmith (1994) built from just :math:`t(t+1)/2` Hadamards and
controlled phases, against :math:`O(t 2^t)` operations for the classical
FFT. Applied to :math:`\sum_x|x\rangle|a^x \bmod N\rangle`, it turns the
period into sharp peaks at multiples of :math:`2^t/r`, which a continued
fraction expansion reads off. This example checks the QFT circuit against
the DFT matrix and factors :math:`15 = 3 \times 5` with :math:`a = 7`.
"""

import matplotlib.pyplot as plt
import numpy as np

from physicskit.quantum.chapters.quantum_circuits import (
    QuantumCircuit,
    modular_exponentiation_circuit,
    qft_circuit,
    shor_period_from_measurement,
)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# %%
# The QFT circuit is the discrete Fourier transform
# -------------------------------------------------
n = 4
U = qft_circuit(n).unitary()
F = np.exp(2j * np.pi * np.outer(np.arange(2**n), np.arange(2**n)) / 2**n) / np.sqrt(2**n)
qft_error = np.abs(U - F).max()
axes[0].imshow(np.angle(U), cmap="twilight", interpolation="nearest")
axes[0].set_title(f"QFT on {n} qubits: phase of $U_{{yx}}$\n|U - DFT| = {qft_error:.1e}", fontsize=10)
axes[0].set_xlabel("input x")
axes[0].set_ylabel("output y")
ts = np.arange(2, 21)
axes[1].semilogy(ts, ts * (ts + 1) / 2, "o-", color="crimson", label=r"QFT gates, $t(t+1)/2$")
axes[1].semilogy(ts, ts * 2.0**ts, "o-", color="gray", label=r"classical FFT, $t\,2^t$")
axes[1].set_xlabel("qubits t")
axes[1].set_ylabel("operations")
axes[1].set_title("Fourier transform cost")
axes[1].legend()

# %%
# Period finding for N = 15, a = 7
# --------------------------------
a, N, t = 7, 15, 8
qc, m = modular_exponentiation_circuit(a, N, t)
qc.append(qft_circuit(t).inverse())
p = QuantumCircuit.probabilities(qc.run(), t + m, tuple(range(t)))
axes[2].bar(np.arange(2**t), p, width=2.0, color="navy")
axes[2].set_xlabel("counting register y")
axes[2].set_ylabel("probability")
counts = QuantumCircuit.sample(p, shots=200, seed=0)
outcomes = np.nonzero(counts)[0]
periods = {int(y): shor_period_from_measurement(int(y), t, N) for y in outcomes}
r = max(periods.values())
factors = sorted({int(np.gcd(a ** (r // 2) - 1, N)), int(np.gcd(a ** (r // 2) + 1, N))})
axes[2].set_title(f"$7^x$ mod 15: peaks at multiples of $2^8/r$, r = {r}, 15 = {factors[0]} x {factors[1]}", fontsize=10)
plt.tight_layout()
plt.show()
print("measured y -> period candidate:", periods)
print(f"period r = {r}; gcd(7^{r // 2} -+ 1, 15) = {factors}")

# %%
# Check
# -----
assert qft_error < 1e-12
np.testing.assert_array_equal(np.nonzero(p > 1e-9)[0], [0, 64, 128, 192])
assert r == 4 and pow(a, r, N) == 1 and factors == [3, 5]
