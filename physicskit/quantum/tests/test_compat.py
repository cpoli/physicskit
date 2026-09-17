"""Test for physicskit.quantum._compat's scipy<1.15 fallback branch.

The dev/CI environment installs scipy>=1.15 (where scipy.special
already has sph_harm_y natively), so the ``except ImportError`` branch
that builds a fallback from the older, deprecated ``sph_harm`` is dead
code under the installed version -- but it is real code protecting
users on scipy>=1.10,<1.15 (the package's declared minimum), so it's
worth exercising directly rather than pragma-excluding it.

This simulates the old-scipy environment by temporarily removing
``scipy.special.sph_harm_y`` and installing a fake ``sph_harm`` with
the old (m, n, phi, theta) signature/convention, then reloading the
compat module so it re-runs its import-time fallback logic, and
restores everything afterward regardless of outcome.
"""

from __future__ import annotations

import importlib

import scipy.special as sp

import physicskit.quantum._compat as compat


def test_sph_harm_y_fallback_matches_native_implementation_when_scipy_is_old():
    real_sph_harm_y = sp.sph_harm_y

    def fake_old_sph_harm(m, n, phi, theta):
        # The old, deprecated scipy.special.sph_harm's argument order and
        # convention: (m, n, azimuthal, polar).
        return real_sph_harm_y(n, m, theta, phi)

    del sp.sph_harm_y
    sp.sph_harm = fake_old_sph_harm
    try:
        importlib.reload(compat)
        n, m, theta, phi = 2, 1, 0.3, 0.7
        assert compat.sph_harm_y(n, m, theta, phi) == real_sph_harm_y(n, m, theta, phi)
    finally:
        del sp.sph_harm
        sp.sph_harm_y = real_sph_harm_y
        importlib.reload(compat)
