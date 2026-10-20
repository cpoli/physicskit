"""
Cross-check of physicskit's FLRW cosmology against Astropy.

Three universes are built in both packages, with no radiation (Astropy's
``Tcmb0=0``): flat Lambda-CDM, an open matter-only universe, and a closed
universe with a cosmological constant. Each package then computes, on its
own, the comoving and luminosity distances on a grid of redshifts and the
age of the universe at several scale factors. physicskit integrates
``quad`` over E(z); Astropy uses its own quadratures and, for the flat
case, its closed-form elliptic-integral solution.

Each comparison is asserted on the relative deviation; ``RESULTS`` holds
the largest deviations for the paper's validation table.
"""

import astropy.units as u
import numpy as np
from astropy.cosmology import FlatLambdaCDM, LambdaCDM

from physicskit.relativity.chapters.cosmology import FLRWCosmology

RESULTS = {}


def record(name, dev, tol):
    RESULTS[name] = float(dev)
    assert dev < tol, (name, dev)
    print(f"{name:60s} max deviation {dev:.1e}")


def rel(a, b):
    return np.max(np.abs(np.asarray(a) / np.asarray(b) - 1))


H0 = 70.0
universes = {
    "flat Lambda-CDM": (0.3, 0.7, FlatLambdaCDM(H0=H0, Om0=0.3, Tcmb0=0.0)),
    "open, matter only": (0.3, 0.0, LambdaCDM(H0=H0, Om0=0.3, Ode0=0.0, Tcmb0=0.0)),
    "closed Lambda-CDM": (0.5, 0.8, LambdaCDM(H0=H0, Om0=0.5, Ode0=0.8, Tcmb0=0.0)),
}
z = np.array([0.01, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
a = np.array([0.1, 0.25, 0.5, 1.0])

dev_dc = dev_dl = dev_age = 0.0
for Om, OL, ref in universes.values():
    pk = FLRWCosmology(H0=H0, Omega_m=Om, Omega_r=0.0, Omega_Lambda=OL)
    dev_dc = max(dev_dc, rel([pk.comoving_distance_mpc(zi) for zi in z], ref.comoving_distance(z).to_value(u.Mpc)))
    dev_dl = max(dev_dl, rel([pk.luminosity_distance_mpc(zi) for zi in z], ref.luminosity_distance(z).to_value(u.Mpc)))
    dev_age = max(dev_age, rel([pk.age_gyr(ai) for ai in a], ref.age(1 / a - 1).to_value(u.Gyr)))
record("FLRW: comoving distance (3 universes, z = 0.01-10)", dev_dc, 1e-10)
record("FLRW: luminosity distance, with curvature", dev_dl, 1e-10)
record("FLRW: age at a = 0.1-1", dev_age, 1e-8)
