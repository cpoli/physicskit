"""
Wall-clock time of physicskit, SciPy, QuTiP and Astropy on shared tasks.

Each case runs in a fresh process with one BLAS thread, so the numbers
compare algorithms, not thread counts. Every case is called once before
it is timed (numba compilation, Astropy's and QuTiP's set-up are not
counted), then timed as the best of three calls. Results go to
``paper/results/benchmarks.json``.

Tasks:
- ode: an eccentric Kepler orbit over N periods with the adaptive
  Dormand-Prince integrator, physicskit's numba ``dopri5_integrate``
  against SciPy's ``solve_ivp(method="RK45")``, at the same tolerances.
- lindblad: a driven, damped cavity of dimension d, 101 output times,
  physicskit's ``solve_lindblad`` (exponential of the Liouvillian) against
  QuTiP's ``mesolve`` (ODE integration).
- spin_chain: the two lowest levels of a periodic transverse-field Ising
  chain of N spins, from building the sparse Hamiltonian to ``eigsh``,
  against QuTiP (``tensor`` products and ``eigenstates(sparse=True)``).
- cosmology: luminosity distances at n redshifts, physicskit's ``quad``
  per redshift against Astropy's vectorized ``luminosity_distance``.

Usage (from the repository root)::

    python paper/benchmarks/run_benchmarks.py [--quick]
"""

import json
import multiprocessing as mp
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[var] = "1"


def ode_case(package, periods):
    import numpy as np

    e = 0.6
    s0 = np.array([1.0 - e, 0.0, 0.0, np.sqrt((1 + e) / (1 - e))])
    t_end, rtol, atol = 2 * np.pi * periods, 1e-9, 1e-12
    if package == "physicskit":
        from numba import njit

        from physicskit.integrators import dopri5_integrate

        @njit
        def kepler(s, t, p):
            r3 = (s[0] ** 2 + s[1] ** 2) ** 1.5
            return np.array([s[2], s[3], -s[0] / r3, -s[1] / r3])

        run = lambda: dopri5_integrate(kepler, s0, 0.0, t_end, 1e-3, np.zeros(0), rtol=rtol, atol=atol, max_steps=10**8)[1][-1]
    else:
        from scipy.integrate import solve_ivp

        def kepler(t, s):
            r3 = (s[0] ** 2 + s[1] ** 2) ** 1.5
            return np.array([s[2], s[3], -s[0] / r3, -s[1] / r3])

        run = lambda: solve_ivp(kepler, (0.0, t_end), s0, method="RK45", rtol=rtol, atol=atol).y[:, -1]
    return measure(run, lambda s: float(np.abs(s - s0).max()))


def lindblad_case(package, dim):
    import numpy as np
    import qutip as qt

    a = qt.destroy(dim)
    H_qt = 0.3 * a.dag() * a + 0.4 * (a + a.dag())
    c_qt = [np.sqrt(0.5) * a]
    t = np.linspace(0.0, 20.0, 101)
    if package == "physicskit":
        from physicskit.quantum.chapters.open_systems import solve_lindblad

        H, c_ops, rho0 = H_qt.full(), [c.full() for c in c_qt], np.eye(dim)[0]
        run = lambda: solve_lindblad(rho0, H, c_ops, t)[-1]
    else:
        run = lambda: qt.mesolve(H_qt, qt.basis(dim, 0), t, c_qt).states[-1].full()
    n = np.diag(np.arange(dim))
    return measure(run, lambda rho: float(np.trace(n @ rho).real))


def spin_chain_case(package, N):
    if package == "physicskit":
        from physicskit.condensed.spin_chains import lowest_eigenstates, tfim_hamiltonian

        run = lambda: lowest_eigenstates(tfim_hamiltonian(N, J=1.0, h=0.7)[0], k=2)[0]
    else:
        import qutip as qt

        def run():
            def site(op, i):
                ops = [qt.qeye(2)] * N
                ops[i] = op
                return qt.tensor(ops)

            sx = [site(qt.sigmax(), i) for i in range(N)]
            H = sum(-sx[i] * sx[(i + 1) % N] - 0.7 * site(qt.sigmaz(), i) for i in range(N))
            return H.eigenenergies(sparse=True, eigvals=2)

    return measure(run, lambda E: float(E[0].real))


def cosmology_case(package, n):
    import numpy as np

    z = np.linspace(0.01, 5.0, n)
    if package == "physicskit":
        from physicskit.relativity.chapters.cosmology import FLRWCosmology

        cosmo = FLRWCosmology(H0=70.0, Omega_m=0.3, Omega_r=0.0, Omega_Lambda=0.7)
        run = lambda: np.array([cosmo.luminosity_distance_mpc(zi) for zi in z])
    else:
        import astropy.units as u
        from astropy.cosmology import FlatLambdaCDM

        cosmo = FlatLambdaCDM(H0=70.0, Om0=0.3, Tcmb0=0.0)
        run = lambda: cosmo.luminosity_distance(z).to_value(u.Mpc)
    return measure(run, lambda d: float(d[-1]))


def measure(run, summary):
    value = summary(run())  # warm-up: compilation, caches
    seconds = []
    for _ in range(3):
        start = time.perf_counter()
        run()
        seconds.append(time.perf_counter() - start)
    return {"seconds": min(seconds), "value": value}


def isolated(func, *args):
    with mp.get_context("spawn").Pool(1) as pool:
        return pool.apply(func, args)


CASES = {
    "ode": (ode_case, "periods", ("physicskit", "scipy"), (1, 10, 100, 1000), (1, 10)),
    "lindblad": (lindblad_case, "dim", ("physicskit", "qutip"), (5, 10, 20, 30, 40, 50), (5, 10)),
    "spin_chain": (spin_chain_case, "N", ("physicskit", "qutip"), (8, 10, 12, 14, 16, 18), (8, 10)),
    "cosmology": (cosmology_case, "n", ("physicskit", "astropy"), (10, 100, 1000, 10000), (10, 100)),
}
TOLERANCE = {"ode": 1e-4, "lindblad": 1e-5, "spin_chain": 1e-9, "cosmology": 1e-6}  # the packages solve the same problem


def main(quick=False):
    import importlib.metadata as metadata

    results = {
        "versions": {p: metadata.version(p) for p in ("physicskit", "qutip", "astropy", "numpy", "scipy", "numba")},
        "platform": platform.platform(),
        "processor": platform.processor(),
        "python": platform.python_version(),
        "threads": 1,
    }
    for task, (case, size, packages, sizes, quick_sizes) in CASES.items():
        rows = results[task] = []
        for value in quick_sizes if quick else sizes:
            for package in packages:
                r = isolated(case, package, value)
                rows.append({"size": value, "package": package, **r})
                print(f"{task:10s} {size}={value:<6d} {package:10s} {r['seconds']:9.4f} s  value {r['value']:.10g}", flush=True)
            a, b = (r["value"] for r in rows[-2:])
            if task == "ode":  # both return to perihelion within the tolerance-limited drift
                assert a < TOLERANCE[task] * value and b < TOLERANCE[task] * value, (task, value, a, b)
            else:
                assert abs(a - b) < TOLERANCE[task] * max(1.0, abs(b)), (task, value, a, b)
    name = "benchmarks_quick.json" if quick else "benchmarks.json"
    out = ROOT / "paper" / "results" / name
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main(quick="--quick" in sys.argv)
