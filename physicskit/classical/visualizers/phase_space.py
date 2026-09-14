"""Phase-space diagnostics: (q, p) portraits, Poincare sections, and the
SO(3) angular-momentum (Casimir) sphere for rigid bodies.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

__all__ = [
    "plot_phase_portrait",
    "plot_phase_swarm",
    "poincare_section",
    "plot_poincare_section",
    "plot_so3_momentum_sphere",
]


def plot_phase_portrait(q: np.ndarray, p: np.ndarray, ax=None, label=None, **kwargs):
    """Plot a single (q, p) trajectory in phase space.

    Parameters
    ----------
    q, p : ndarray
        1D (a single degree of freedom over time) or 2D
        (``(n_steps, ndof)``, in which case each column is drawn as
        its own curve).
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    label : str, optional
        Legend label (per-DOF suffix added automatically for 2D input).
    **kwargs
        Forwarded to ``ax.plot``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    q = np.asarray(q)
    p = np.asarray(p)
    if q.ndim == 1:
        ax.plot(q, p, label=label, **kwargs)
    else:
        for i in range(q.shape[1]):
            ax.plot(q[:, i], p[:, i], label=f"{label} dof {i}" if label else f"dof {i}", **kwargs)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("Phase portrait")
    if label:
        ax.legend()
    return ax


def plot_phase_swarm(q: np.ndarray, p: np.ndarray, ax=None, s=4, **kwargs):
    """Scatter an ensemble of phase points.

    E.g. a :class:`physicskit.classical.systems.hamiltonian.PendulumSwarm` snapshot,
    the standard way to visualize Liouville-theorem shearing of a
    phase-space patch.

    Parameters
    ----------
    q, p : ndarray, shape (n,)
        Phase-point positions and momenta.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    s : float
        Marker size, forwarded to ``ax.scatter``.
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(q, p, s=s, **kwargs)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("Phase-space swarm")
    return ax


def poincare_section(q: np.ndarray, p: np.ndarray, section_q: np.ndarray, section_p: np.ndarray, value: float = 0.0, direction: int = 1):
    """Compute Poincare-section crossing points.

    Records ``(q, p)`` each time ``section_q`` crosses ``value`` moving
    in the sign given by ``direction`` (+1 for increasing, -1 for
    decreasing), linearly interpolating between the bracketing samples
    for sub-step accuracy. Typical usage for Henon-Heiles: pass y as
    ``section_q``, take the section at y=0 with py>0 (direction=+1),
    and record (x, px) crossings.

    Parameters
    ----------
    q, p : ndarray
        Coordinates to record at each crossing.
    section_q, section_p : ndarray
        The coordinate (and its conjugate momentum) whose crossings of
        ``value`` define the section; same length as ``q``/``p``.
    value : float
        Crossing threshold.
    direction : {1, -1}
        +1 to record crossings where ``section_q`` is increasing
        through ``value``, -1 for decreasing.

    Returns
    -------
    crossings_q, crossings_p : ndarray
        Interpolated ``(q, p)`` at each crossing.
    """
    section_q = np.asarray(section_q)
    section_p = np.asarray(section_p)
    crossings_q = []
    crossings_p = []
    for i in range(len(section_q) - 1):
        y0, y1 = section_q[i] - value, section_q[i + 1] - value
        crosses = (y0 <= 0 < y1) if direction > 0 else (y0 >= 0 > y1)
        if not crosses:
            continue
        if section_p[i] * direction < 0:
            continue
        frac = -y0 / (y1 - y0) if y1 != y0 else 0.0
        crossings_q.append(q[i] + frac * (q[i + 1] - q[i]))
        crossings_p.append(p[i] + frac * (p[i + 1] - p[i]))
    return np.array(crossings_q), np.array(crossings_p)


def plot_poincare_section(q, p, section_q, section_p, value=0.0, direction=1, ax=None, **kwargs):
    """Scatter-plot the Poincare-section crossings of a trajectory.

    Parameters
    ----------
    q, p : ndarray
        Coordinates to record at each crossing.
    section_q, section_p : ndarray
        The coordinate (and its conjugate momentum) whose crossings of
        ``value`` define the section.
    value : float
        Crossing threshold.
    direction : {1, -1}
        Crossing direction; see :func:`poincare_section`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw on; a new figure is created if omitted.
    **kwargs
        Forwarded to ``ax.scatter``.

    Returns
    -------
    matplotlib.axes.Axes
    """
    cq, cp = poincare_section(q, p, section_q, section_p, value=value, direction=direction)
    if ax is None:
        _, ax = plt.subplots()
    ax.scatter(cq, cp, s=2, **kwargs)
    ax.set_xlabel("q")
    ax.set_ylabel("p")
    ax.set_title("Poincare section")
    return ax


def plot_so3_momentum_sphere(omega_trajectory: np.ndarray, I1: float, I2: float, I3: float, ax=None, n_grid: int = 40):
    """Plot a rigid body's body-frame angular-momentum trajectory on the Casimir sphere.

    ``L(t) = (I1*w1, I2*w2, I3*w3)`` on the SO(3) Casimir sphere
    ``|L| = |L(0)|``.

    The wireframe sphere shown is the constant-``|L|`` Casimir surface; the
    trajectory is additionally constrained to the (generally
    non-spherical) constant-energy ellipsoid, so the two surfaces'
    intersection traces out the actual admissible polhode curves --
    this is the geometric picture behind the Intermediate Axis Theorem.

    Parameters
    ----------
    omega_trajectory : ndarray, shape (n_steps, 3)
        Body-frame angular velocity trajectory, e.g.
        ``result.y[:, :3]`` from integrating an
        :class:`~physicskit.classical.systems.rotations.EulerTop`.
    I1, I2, I3 : float
        Principal moments of inertia.
    ax : mpl_toolkits.mplot3d.Axes3D, optional
        Axes to draw on; a new 3D figure is created if omitted.
    n_grid : int
        Resolution of the wireframe sphere.

    Returns
    -------
    mpl_toolkits.mplot3d.Axes3D
    """
    omega_trajectory = np.asarray(omega_trajectory)
    L = np.column_stack([I1 * omega_trajectory[:, 0], I2 * omega_trajectory[:, 1], I3 * omega_trajectory[:, 2]])
    radius = np.linalg.norm(L[0])

    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

    u = np.linspace(0, 2 * np.pi, n_grid)
    v = np.linspace(0, np.pi, n_grid)
    xs = radius * np.outer(np.cos(u), np.sin(v))
    ys = radius * np.outer(np.sin(u), np.sin(v))
    zs = radius * np.outer(np.ones_like(u), np.cos(v))
    ax.plot_wireframe(xs, ys, zs, color="lightgray", linewidth=0.3, alpha=0.5)

    ax.plot(L[:, 0], L[:, 1], L[:, 2], color="crimson", linewidth=1.2)
    ax.set_xlabel("L1")
    ax.set_ylabel("L2")
    ax.set_zlabel("L3")
    ax.set_title("Body-frame angular momentum on the Casimir sphere")
    return ax
