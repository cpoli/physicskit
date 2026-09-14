"""Eigenvalue support of a truncated random unitary matrix (see
``physicskit.rmt.ensembles.truncated_unitary`` for the construction and the
numerical verification behind the support-radius formula below).

Reference: K. Zyczkowski, H.-J. Sommers, J. Phys. A 33 (2000) 2045.
"""

import numpy as np


def truncated_unitary_edge_radius(alpha: float) -> float:
    """Support radius of a truncated unitary matrix's eigenvalues:
    sqrt(alpha), for alpha = m/n the fraction of the full unitary matrix
    kept.

    Verified numerically (not assumed from memory -- an initial guess
    of sqrt(1-alpha) had exactly the wrong alpha-dependence, caught
    immediately since it would shrink toward 0 as more of the unitary
    matrix is kept, contradicting the alpha=1 exact-unit-circle limit):
    the 99th-percentile empirical eigenvalue radius converges cleanly to
    this value as n grows at fixed alpha -- see
    ``physicskit.rmt.ensembles.truncated_unitary`` module docstring and
    ``tests/test_truncated_unitary.py``.

    Precision note: this gives only the support EDGE, not the (known to
    be non-uniform, edge-weighted) bulk density within the disk -- see
    module docstring for what was and wasn't verified.

    Parameters
    ----------
    alpha : float
        Fraction of the full unitary matrix kept, in (0, 1].

    Returns
    -------
    float
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError(f"alpha must be in (0, 1], got {alpha}")
    return float(np.sqrt(alpha))
