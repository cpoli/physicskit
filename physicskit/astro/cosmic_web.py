r"""The Zel'dovich (1970) approximation and the formation of the cosmic web.

In an expanding universe, small density perturbations do not stay small
forever: gravity amplifies them until they collapse along their fastest-
contracting direction first, producing the sheets ("pancakes"), filaments,
and dense nodes that together make up the observed large-scale structure of
the universe -- the *cosmic web*. Zel'dovich (1970, *Astronomy and
Astrophysics* 5, "Gravitational instability: An approximate theory for
large density perturbations") gave the classic, mathematically clean
description of this process: to first order in the displacement of matter
away from a uniform background, each fluid element simply moves in a
straight line (in comoving coordinates) at a rate set by the local gradient
of a fixed potential. The approximation is exact in one dimension and
remains an excellent guide to the onset of nonlinear structure in more
dimensions, which is why it is still the standard way to generate the
initial conditions of cosmological N-body simulations today.

This module works in a flat, matter-dominated (Einstein-de Sitter)
background, where the scale factor grows as :math:`a(t)\propto t^{2/3}`
and the linear growth factor of density perturbations is proportional to
the scale factor, :math:`D(t)\propto a(t)`. Rather than tracking time
directly, every function here is parametrized by the growth factor
:math:`D` itself, which increases monotonically from :math:`D=0` at the
initial (unperturbed) time.

Given a fixed Lagrangian ("unperturbed") position :math:`\mathbf{q}` and a
scalar displacement potential :math:`\Phi(\mathbf{q})`, the Zel'dovich
approximation maps :math:`\mathbf{q}` to a comoving Eulerian position

.. math::

    \mathbf{x}(\mathbf{q}, D) = \mathbf{q} - D\,\nabla\Phi(\mathbf{q}),

i.e. every fluid element streams away from its starting point along a
straight line, in the fixed direction :math:`-\nabla\Phi(\mathbf{q})`, at a
rate that grows linearly with :math:`D`. This module builds :math:`\Phi` as
a sum of a modest number of plane waves -- a simple stand-in for a
realistic Gaussian random field, exactly as used pedagogically --

.. math::

    \Phi(\mathbf{q}) = \sum_{n=1}^{N} A_n\cos(\mathbf{k}_n\cdot\mathbf{q}+\varphi_n),

with random wavevectors :math:`\mathbf{k}_n`, amplitudes :math:`A_n`, and
phases :math:`\varphi_n`. Because mass is conserved as fluid elements move,
the local density contrast follows directly from the Jacobian of the
mapping :math:`\mathbf{q}\mapsto\mathbf{x}`:

.. math::

    \frac{\rho(\mathbf{x})}{\bar\rho} = \frac{1}{\left|\det\left(I - D\,H_\Phi(\mathbf{q})\right)\right|},

where :math:`H_\Phi` is the Hessian of :math:`\Phi`. This formally
diverges -- a caustic, i.e. a "pancake" -- the first time :math:`D` reaches
:math:`1/\lambda_{\max}(\mathbf{q})`, where :math:`\lambda_{\max}(\mathbf{q})`
is the largest eigenvalue of :math:`H_\Phi` at that point. The very first
caustic anywhere in the field therefore forms at

.. math::

    D_{\rm collapse} = \frac{1}{\max_{\mathbf{q}}\lambda_{\max}(\mathbf{q})}.

For a single plane-wave potential this is exact and closed-form
(:math:`D_{\rm collapse}=1/(A|\mathbf{k}|^2)`), which makes it a clean,
directly testable prediction; see the test suite for the exact derivation.

- :func:`random_displacement_potential` -- draw a random multi-mode
  displacement potential :math:`\Phi`.
- :func:`zeldovich_displacement` -- the analytic gradient
  :math:`\nabla\Phi(\mathbf{q})`.
- :func:`zeldovich_position` -- the Zel'dovich mapping
  :math:`\mathbf{x}(\mathbf{q}, D)`.
- :func:`zeldovich_hessian_eigenvalues` -- the eigenvalues of
  :math:`H_\Phi(\mathbf{q})`, which control local collapse.
- :func:`first_caustic_time` -- the growth factor at which the first
  pancake forms anywhere in the field.
- :func:`lagrangian_grid` -- a regular grid of Lagrangian tracer particles
  to evolve under the mapping.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "random_displacement_potential",
    "zeldovich_displacement",
    "zeldovich_position",
    "zeldovich_hessian_eigenvalues",
    "first_caustic_time",
    "lagrangian_grid",
]


def random_displacement_potential(n_modes, k_min, k_max, amplitude_scale, seed=None, ndim=2):
    r"""Draw a random multi-mode displacement potential :math:`\Phi(\mathbf{q})`.

    Builds :math:`\Phi(\mathbf{q}) = \sum_{n=1}^{N}A_n\cos(\mathbf{k}_n\cdot
    \mathbf{q}+\varphi_n)` -- a simple stand-in for a realistic Gaussian
    random field -- with wavevectors :math:`\mathbf{k}_n` of random
    orientation and magnitude drawn uniformly from :math:`[k_{\min},
    k_{\max}]`, amplitudes drawn uniformly from :math:`[0,
    \text{amplitude\_scale}]`, and phases drawn uniformly from
    :math:`[0, 2\pi)`.

    Parameters
    ----------
    n_modes : int
        Number of plane-wave modes, :math:`N`.
    k_min, k_max : float
        Lower and upper bounds on the wavenumber magnitude
        :math:`|\mathbf{k}_n|`, setting the band of scales structure forms
        at.
    amplitude_scale : float
        Upper bound on each mode's amplitude :math:`A_n`.
    seed : int, optional
        Random seed, for reproducibility.
    ndim : int, default=2
        Spatial dimension. Only ``ndim=2`` is supported; the analytic
        gradient/Hessian machinery in this module is 2D-specific.

    Returns
    -------
    k_vectors : ndarray, shape (n_modes, ndim)
        Wavevectors :math:`\mathbf{k}_n`.
    amplitudes : ndarray, shape (n_modes,)
        Amplitudes :math:`A_n`.
    phases : ndarray, shape (n_modes,)
        Phases :math:`\varphi_n`, in :math:`[0, 2\pi)`.

    Raises
    ------
    ValueError
        If ``ndim != 2``, ``n_modes < 1``, or ``k_max < k_min``.

    Examples
    --------
    >>> k_vectors, amplitudes, phases = random_displacement_potential(8, k_min=1.0, k_max=3.0, amplitude_scale=0.1, seed=0)
    >>> k_vectors.shape, amplitudes.shape, phases.shape
    ((8, 2), (8,), (8,))
    >>> bool(np.all(amplitudes >= 0.0) and np.all(amplitudes <= 0.1))
    True
    """
    if ndim != 2:
        raise ValueError("random_displacement_potential only supports ndim=2.")
    if n_modes < 1:
        raise ValueError("n_modes must be >= 1.")
    if k_max < k_min:
        raise ValueError("k_max must be >= k_min.")

    rng = np.random.default_rng(seed)
    k_mag = rng.uniform(k_min, k_max, n_modes)
    angle = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    k_vectors = np.stack([k_mag * np.cos(angle), k_mag * np.sin(angle)], axis=-1)
    amplitudes = rng.uniform(0.0, amplitude_scale, n_modes)
    phases = rng.uniform(0.0, 2.0 * np.pi, n_modes)
    return k_vectors, amplitudes, phases


def _phase_argument(q, k_vectors, phases):
    """``k_n . q_i + phase_n`` for every particle ``i`` and mode ``n``, shape (n_particles, n_modes)."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    k_vectors = np.asarray(k_vectors, dtype=float)
    phases = np.asarray(phases, dtype=float)
    return q @ k_vectors.T + phases[None, :]


def zeldovich_displacement(q, k_vectors, amplitudes, phases):
    r"""The analytic displacement field :math:`\nabla\Phi(\mathbf{q})`.

    .. math::

        \nabla\Phi(\mathbf{q}) = \sum_n -A_n\,\mathbf{k}_n\,
        \sin(\mathbf{k}_n\cdot\mathbf{q}+\varphi_n)

    Vectorized over all particles and modes at once (no explicit loops).

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian (unperturbed) positions.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Potential parameters, e.g. from :func:`random_displacement_potential`.

    Returns
    -------
    ndarray, shape (n_particles, 2)
        :math:`\nabla\Phi` evaluated at each ``q``.

    Examples
    --------
    >>> import numpy as np
    >>> k_vectors = np.array([[1.0, 0.0]])
    >>> amplitudes = np.array([2.0])
    >>> phases = np.array([np.pi / 2])  # sin(pi/2) = 1
    >>> grad = zeldovich_displacement(np.array([[0.0, 0.0]]), k_vectors, amplitudes, phases)
    >>> np.round(grad + 0.0, 6)  # "+ 0.0" avoids a printed "-0." for exact zeros
    array([[-2.,  0.]])
    """
    k_vectors = np.asarray(k_vectors, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    theta = _phase_argument(q, k_vectors, phases)
    weighted_sin = np.sin(theta) * amplitudes[None, :]
    return -(weighted_sin @ k_vectors)


def zeldovich_position(q, D, k_vectors, amplitudes, phases):
    r"""The Zel'dovich mapping :math:`\mathbf{x}(\mathbf{q}, D) = \mathbf{q} - D\,\nabla\Phi(\mathbf{q})`.

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian (unperturbed) positions.
    D : float
        Linear growth factor (dimensionless "time"), :math:`D=0` at the
        initial, unperturbed time.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Potential parameters, e.g. from :func:`random_displacement_potential`.

    Returns
    -------
    ndarray, shape (n_particles, 2)
        Comoving Eulerian positions :math:`\mathbf{x}`.

    See Also
    --------
    zeldovich_displacement : The gradient field used here.

    Examples
    --------
    >>> import numpy as np
    >>> q = np.array([[0.3, 0.7], [1.1, -0.4]])
    >>> k_vectors, amplitudes, phases = random_displacement_potential(4, 1.0, 2.0, 0.1, seed=1)
    >>> x0 = zeldovich_position(q, 0.0, k_vectors, amplitudes, phases)
    >>> bool(np.allclose(x0, q))
    True
    """
    q = np.atleast_2d(np.asarray(q, dtype=float))
    grad_phi = zeldovich_displacement(q, k_vectors, amplitudes, phases)
    return q - D * grad_phi


def zeldovich_hessian_eigenvalues(q, k_vectors, amplitudes, phases):
    r"""Eigenvalues of the analytic Hessian :math:`H_\Phi(\mathbf{q})`.

    The Hessian components follow directly from differentiating the
    plane-wave sum twice,

    .. math::

        \frac{\partial^2\Phi}{\partial q_i\partial q_j}
        = \sum_n -A_n\,k_{n,i}k_{n,j}\cos(\mathbf{k}_n\cdot\mathbf{q}+\varphi_n),

    and its two eigenvalues follow from the closed-form 2x2 symmetric
    formula

    .. math::

        \lambda_\pm = \frac{(\Phi_{xx}+\Phi_{yy}) \pm
        \sqrt{(\Phi_{xx}-\Phi_{yy})^2+4\Phi_{xy}^2}}{2},

    computed directly as arrays (no per-particle ``numpy.linalg.eigh``
    calls).

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian (unperturbed) positions.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Potential parameters, e.g. from :func:`random_displacement_potential`.

    Returns
    -------
    ndarray, shape (n_particles, 2)
        The two eigenvalues of :math:`H_\Phi` at each ``q``, sorted
        ascending along the last axis.

    See Also
    --------
    first_caustic_time : The growth factor at which the largest of these
        eigenvalues, maximized over all ``q``, first causes a caustic.

    Examples
    --------
    >>> import numpy as np
    >>> k_vectors = np.array([[2 * np.pi, 0.0]])
    >>> amplitudes = np.array([1.0])
    >>> phases = np.array([np.pi])  # cos(phase) = -1 at q=0: the compression maximum
    >>> eigs = zeldovich_hessian_eigenvalues(np.array([[0.0, 0.0]]), k_vectors, amplitudes, phases)
    >>> np.round(eigs, 6)
    array([[ 0.      , 39.478418]])
    """
    k_vectors = np.asarray(k_vectors, dtype=float)
    amplitudes = np.asarray(amplitudes, dtype=float)
    theta = _phase_argument(q, k_vectors, phases)
    weighted_cos = np.cos(theta) * amplitudes[None, :]

    kx = k_vectors[:, 0]
    ky = k_vectors[:, 1]
    Pxx = -(weighted_cos @ (kx * kx))
    Pyy = -(weighted_cos @ (ky * ky))
    Pxy = -(weighted_cos @ (kx * ky))

    trace = Pxx + Pyy
    disc = np.sqrt((Pxx - Pyy) ** 2 + 4.0 * Pxy**2)
    lambda_lo = 0.5 * (trace - disc)
    lambda_hi = 0.5 * (trace + disc)
    return np.stack([lambda_lo, lambda_hi], axis=-1)


def first_caustic_time(q, k_vectors, amplitudes, phases):
    r"""The growth factor :math:`D_{\rm collapse}` at which the first caustic forms.

    .. math::

        D_{\rm collapse} = \frac{1}{\max_{\mathbf{q}}\lambda_{\max}(\mathbf{q})}

    evaluated over the sampled points ``q``. The sampled maximum
    eigenvalue can only underestimate the maximum over the *continuous*
    field, so this is an upper bound on the true collapse time that
    tightens toward it as ``q`` samples the field more finely.

    Parameters
    ----------
    q : ndarray, shape (n_particles, 2)
        Lagrangian (unperturbed) sample points, e.g. from
        :func:`lagrangian_grid`.
    k_vectors : ndarray, shape (n_modes, 2)
    amplitudes : ndarray, shape (n_modes,)
    phases : ndarray, shape (n_modes,)
        Potential parameters, e.g. from :func:`random_displacement_potential`.

    Returns
    -------
    float
        :math:`D_{\rm collapse}`, the growth factor at which the first
        pancake forms anywhere among the sampled points.

    See Also
    --------
    zeldovich_hessian_eigenvalues : Supplies the per-point eigenvalues
        this is maximized over.

    Examples
    --------
    >>> import numpy as np
    >>> k_vectors = np.array([[2 * np.pi, 0.0]])
    >>> amplitudes = np.array([1.0])
    >>> phases = np.array([np.pi])
    >>> round(first_caustic_time(np.array([[0.0, 0.0]]), k_vectors, amplitudes, phases), 6)
    0.02533
    """
    eigs = zeldovich_hessian_eigenvalues(q, k_vectors, amplitudes, phases)
    return float(1.0 / np.max(eigs[:, -1]))


def lagrangian_grid(n_side, box_size):
    r"""A regular grid of Lagrangian tracer particles over :math:`[0, L)^2`.

    The standard starting point for a Zel'dovich visualization: an
    initially perfectly uniform grid of tracer particles, whose progressive
    deformation into filaments, sheets, and nodes under
    :func:`zeldovich_position` is itself part of what makes the
    visualization compelling.

    Parameters
    ----------
    n_side : int
        Number of grid points along each axis.
    box_size : float
        Side length :math:`L` of the (periodic) comoving box.

    Returns
    -------
    ndarray, shape (n_side * n_side, 2)
        Lagrangian positions ``q``, tiling :math:`[0, L)^2` on a regular
        grid.

    Examples
    --------
    >>> q = lagrangian_grid(4, 1.0)
    >>> q.shape
    (16, 2)
    >>> bool(np.all((q >= 0.0) & (q < 1.0)))
    True
    """
    coords = np.linspace(0.0, box_size, n_side, endpoint=False)
    qx, qy = np.meshgrid(coords, coords, indexing="ij")
    return np.stack([qx.ravel(), qy.ravel()], axis=-1)
