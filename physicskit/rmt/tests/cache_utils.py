"""Disk cache for eigenvalue (and eigenvector, when present) data
produced by sampling a MatrixEnsemble in tests -- avoids re-running
expensive matrix diagonalization every time the test suite reruns.

Usage:
    spectrum = cached_sample(ensemble, n_samples=30)

instead of:
    spectrum = ensemble.sample(n_samples=30)

Cache key is derived from the ensemble's class name, its public
constructor-set attributes (n, m, m1, m2, beta, seed, ...), and
n_samples -- so a fresh cache entry is created whenever any of those
change, and reruns with identical parameters load straight from disk.

NOT used for reproducibility tests that compare two independently
constructed ensemble instances with the same seed (e.g.
``ens_a.sample(...)`` vs ``ens_b.sample(...)``): caching by
(class, params, seed) would make both calls hit the same cache entry,
so the second would just re-read the first's cached array rather than
genuinely resampling -- silently defeating the point of that test. Call
``ensemble.sample(...)`` directly there instead.
"""

import hashlib
import json
from pathlib import Path

import numpy as np

from physicskit.rmt.spectrum import Spectrum

CACHE_DIR = Path(__file__).parent / ".physicskit.rmt_test_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _public_params(ensemble):
    """Public, JSON-serializable constructor-set attributes of an
    ensemble instance (n, m, m1, m2, beta, seed, ...) -- excludes
    private attributes like ``_rng``."""
    params = {}
    for key, value in vars(ensemble).items():
        if key.startswith("_"):
            continue
        if isinstance(value, (int, float, str, bool)) or value is None:
            params[key] = value
    return params


def _cache_key(ensemble, n_samples):
    payload = json.dumps(
        {
            "cls": type(ensemble).__name__,
            "params": _public_params(ensemble),
            "n_samples": n_samples,
        },
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:24]


def cached_sample(ensemble, n_samples):
    """Sample ``ensemble`` (or load a matching cached result from disk).

    Parameters
    ----------
    ensemble : physicskit.rmt.ensembles.base.MatrixEnsemble
        An already-constructed ensemble instance (construction itself is
        cheap; only ``.sample()`` -- the diagonalization -- is cached).
    n_samples : int

    Returns
    -------
    Spectrum
    """
    key = _cache_key(ensemble, n_samples)
    path = CACHE_DIR / f"{key}.npz"

    if path.exists():
        data = np.load(path, allow_pickle=False)
        eigenvalues = data["eigenvalues"]
        beta_raw = data["beta"]
        beta = None if np.isnan(beta_raw) else float(beta_raw)
        spectrum = Spectrum(
            eigenvalues=eigenvalues,
            n=int(data["n"]),
            beta=beta,
            ensemble=str(data["ensemble_name"]),
            scale=float(data["scale"]),
        )
        if "eigenvectors" in data.files:
            spectrum.eigenvectors = data["eigenvectors"]
        return spectrum

    spectrum = ensemble.sample(n_samples=n_samples)
    save_kwargs = dict(
        eigenvalues=spectrum.eigenvalues,
        n=spectrum.n,
        beta=np.nan if spectrum.beta is None else float(spectrum.beta),
        ensemble_name=spectrum.ensemble,
        scale=spectrum.scale,
    )
    eigenvectors = getattr(spectrum, "eigenvectors", None)
    if eigenvectors is not None:
        save_kwargs["eigenvectors"] = eigenvectors
    np.savez(path, **save_kwargs)
    return spectrum
