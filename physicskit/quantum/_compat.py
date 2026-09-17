"""Small compatibility shims.

numpy>=2.0 renamed trapz to trapezoid; scipy>=1.15 added sph_harm_y and
deprecated the older sph_harm, so on an installed scipy<1.15 (dependencies
only require scipy>=1.10) we fall back to sph_harm with its swapped
argument order and angle convention.
"""

import numpy as np

trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

try:
    from scipy.special import sph_harm_y as sph_harm_y
except ImportError:  # scipy < 1.15
    from scipy.special import sph_harm as _sph_harm

    def sph_harm_y(n, m, theta, phi):
        """``sph_harm_y(n, m, theta, phi)`` with theta=polar, phi=azimuthal."""
        return _sph_harm(m, n, phi, theta)
