"""Plotting helpers: band structures, Berry curvature heatmaps, edge-state densities, Fermi surfaces.

Matplotlib is used for 2D plots (band structures, Berry-curvature heatmaps,
real-space wavefunction density), and Plotly for interactive 3D Fermi-surface
isosurfaces. Every function returns its figure object rather than calling
``show()``, so it composes cleanly into larger figures or headless pipelines.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go

__all__ = [
    "plot_band_structure",
    "plot_berry_curvature",
    "plot_edge_state_density",
    "plot_lattice_structure",
    "plot_fermi_surface_3d",
]


def plot_band_structure(hamiltonian_func, path_points, labels=None, n_per_segment: int = 100, ax=None):
    """Plot bands along a piecewise-linear path through the Brillouin zone.

    Parameters
    ----------
    hamiltonian_func : callable
        A function ``H(k1, k2)`` returning the Bloch Hamiltonian.
    path_points : sequence of tuple(float, float)
        High-symmetry points (e.g. :math:`\\Gamma, K, M, \\Gamma`) in reduced
        crystal momentum, connected by straight segments.
    labels : sequence of str, optional
        Tick labels for each point in ``path_points``.
    n_per_segment : int, default=100
        Number of sampled k-points per segment.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import graphene_hamiltonian
    >>> path = [(0, 0), (2 * np.pi / 3, 4 * np.pi / 3), (2 * np.pi / 3, 2 * np.pi / 3), (0, 0)]
    >>> fig, ax = plot_band_structure(lambda k1, k2: graphene_hamiltonian(k1, k2), path, labels=["G", "K", "M", "G"])
    >>> isinstance(fig, plt.Figure)
    True
    >>> sum(1 for line in ax.lines if line.get_color() == "C0")
    2
    """
    path_points = [np.asarray(p, dtype=float) for p in path_points]
    ks, distance, ticks = [], [0.0], [0.0]
    for p0, p1 in zip(path_points[:-1], path_points[1:], strict=True):
        for s in np.linspace(0, 1, n_per_segment, endpoint=False):
            ks.append(p0 + s * (p1 - p0))
        seg_len = np.linalg.norm(p1 - p0)
        distance.append(distance[-1] + seg_len)
        ticks.append(distance[-1])
    ks.append(path_points[-1])

    d = [0.0]
    for k0, k1 in zip(ks[:-1], ks[1:], strict=True):
        d.append(d[-1] + np.linalg.norm(k1 - k0))

    energies = np.array([np.linalg.eigvalsh(np.asarray(hamiltonian_func(*k), dtype=complex)) for k in ks])

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    for band in range(energies.shape[1]):
        ax.plot(d, energies[:, band], color="C0")
    for t in ticks:
        ax.axvline(t, color="gray", linewidth=0.5)
    if labels is not None:
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
    ax.set_ylabel("Energy")
    return fig, ax


def plot_berry_curvature(curvature: np.ndarray, ax=None):
    """Plot a Berry-curvature field over the Brillouin zone as a heatmap.

    Parameters
    ----------
    curvature : ndarray, shape (grid_size, grid_size)
        Per-plaquette Berry curvature, as returned by
        :func:`physicskit.condensed.topology.compute_berry_curvature`.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import haldane_model
    >>> from physicskit.condensed.topology import compute_berry_curvature
    >>> H = lambda k1, k2: haldane_model(k1, k2, phi=np.pi / 2)
    >>> F = compute_berry_curvature(H, grid_size=20, band_index=0)
    >>> fig, ax = plot_berry_curvature(F)
    >>> isinstance(fig, plt.Figure)
    True
    """
    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    im = ax.imshow(curvature.T, origin="lower", extent=[0, 2 * np.pi, 0, 2 * np.pi], aspect="auto", cmap="RdBu_r")
    fig.colorbar(im, ax=ax, label="Berry curvature")
    ax.set_xlabel("$k_1$")
    ax.set_ylabel("$k_2$")
    return fig, ax


def plot_edge_state_density(ribbon_hamiltonian_func, k_parallel, ax=None):
    """Plot the real-space density of the mid-gap eigenstate closest to zero energy.

    Useful for visualizing the exponential localization
    :math:`|\\psi(x)|^2 \\sim e^{-2x/\\xi}` of topological edge states exposed
    by :func:`physicskit.condensed.tight_binding.build_ribbon`.

    Parameters
    ----------
    ribbon_hamiltonian_func : callable
        A function ``H(k_parallel)`` returning the finite ribbon Hamiltonian,
        as produced by :func:`~physicskit.condensed.tight_binding.build_ribbon`.
    k_parallel : array_like
        Momentum along the periodic direction(s) at which to evaluate the ribbon.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes
    density : ndarray
        Per-site probability density of the closest-to-zero eigenstate.

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import ssh_lattice_hamiltonian
    >>> from physicskit.condensed.tight_binding import build_ribbon
    >>> H = ssh_lattice_hamiltonian(v=0.5, w=1.0)
    >>> H_wire = build_ribbon(H, open_direction=0, n_cells=15)
    >>> fig, ax, density = plot_edge_state_density(H_wire, k_parallel=[])
    >>> bool(density[0] > density[len(density) // 2])
    True
    """
    H = ribbon_hamiltonian_func(k_parallel)
    eigenvalues, eigenvectors = np.linalg.eigh(H)
    idx = int(np.argmin(np.abs(eigenvalues)))
    psi = eigenvectors[:, idx]
    density = np.abs(psi) ** 2

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure
    ax.plot(np.arange(len(density)), density, marker="o")
    ax.set_xlabel("Site index")
    ax.set_ylabel(r"$|\psi(x)|^2$")
    ax.set_title(f"Closest-to-zero eigenstate, E={eigenvalues[idx]:.4f}")
    return fig, ax, density


def plot_lattice_structure(positions, bonds, weights=None, ax=None, cmap: str = "viridis", bond_linewidth_scale: float = 3.0):
    """Draw a finite tight-binding cluster's real-space structure: atoms and bonds.

    Renders the actual geometry of a finite lattice -- e.g. the output of
    :func:`physicskit.condensed.tight_binding.build_finite_cluster` -- rather
    than an abstract site-index plot: bonds as line segments (linewidth
    proportional to hopping strength, so e.g. SSH's alternating strong/weak
    dimerization is directly visible), atoms as markers optionally colored
    and sized by a per-site weight such as edge-state probability density.

    Parameters
    ----------
    positions : array_like, shape (n_sites, dim)
        Cartesian coordinates of each site, ``dim`` in ``{1, 2}``.
    bonds : sequence of tuple(int, int, complex)
        Real-space hoppings ``(i, j, amplitude)`` indexing into ``positions``.
    weights : array_like, shape (n_sites,), optional
        Per-site scalar (e.g. :math:`|\\psi|^2`) mapped to marker color and
        size. Defaults to uniform, unweighted markers.
    ax : matplotlib.axes.Axes, optional
        Axes to draw into; a new figure is created if omitted.
    cmap : str, default="viridis"
        Colormap used when ``weights`` is given.
    bond_linewidth_scale : float, default=3.0
        Bonds are drawn with linewidth ``bond_linewidth_scale * |amplitude|
        / max(|amplitude|)``.

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : matplotlib.axes.Axes

    Examples
    --------
    >>> import numpy as np
    >>> from physicskit.condensed.models import ssh_lattice_hamiltonian
    >>> from physicskit.condensed.tight_binding import build_finite_cluster
    >>> H, positions, bonds = build_finite_cluster(ssh_lattice_hamiltonian(v=0.5, w=1.0), n_cells=3)
    >>> _, states = np.linalg.eigh(H)
    >>> density = np.abs(states[:, 0]) ** 2
    >>> fig, ax = plot_lattice_structure(positions, bonds, weights=density)
    >>> isinstance(fig, plt.Figure)
    True
    """
    positions = np.atleast_2d(np.asarray(positions, dtype=float))
    x = positions[:, 0]
    y = positions[:, 1] if positions.shape[1] > 1 else np.zeros(len(positions))

    if ax is None:
        fig, ax = plt.subplots()
    else:
        fig = ax.figure

    amplitudes = np.array([abs(amp) for _, _, amp in bonds])
    max_amp = amplitudes.max() if amplitudes.size else 1.0
    for (i, j, _amp), a in zip(bonds, amplitudes, strict=True):
        lw = bond_linewidth_scale * a / max_amp if max_amp > 0 else bond_linewidth_scale
        ax.plot(x[[i, j]], y[[i, j]], color="gray", linewidth=lw, zorder=1, solid_capstyle="round")

    if weights is None:
        ax.scatter(x, y, s=80, color="C0", edgecolor="k", linewidth=0.6, zorder=2)
    else:
        weights = np.asarray(weights, dtype=float)
        max_w = weights.max()
        sizes = 30 + 300 * weights / max_w if max_w > 0 else np.full_like(weights, 30.0)
        sc = ax.scatter(x, y, s=sizes, c=weights, cmap=cmap, edgecolor="k", linewidth=0.6, zorder=2)
        fig.colorbar(sc, ax=ax, label="weight")

    ax.set_aspect("equal")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    return fig, ax


def plot_fermi_surface_3d(dispersion_func, mu: float = 0.0, grid_size: int = 40, k_range: float = np.pi):
    """Render a 3D Fermi-surface isosurface :math:`\\varepsilon(\\mathbf{k}) = \\mu`.

    Parameters
    ----------
    dispersion_func : callable
        A function ``eps(kx, ky, kz)`` returning the band energy.
    mu : float, default=0.0
        Chemical potential (Fermi energy) defining the isosurface.
    grid_size : int, default=40
        Number of samples along each of :math:`k_x, k_y, k_z`.
    k_range : float, default=pi
        Half-width of the cubic sampling box, ``[-k_range, k_range]``.

    Returns
    -------
    plotly.graph_objects.Figure
        A figure containing a single ``Isosurface`` trace.

    Examples
    --------
    >>> import numpy as np
    >>> eps = lambda kx, ky, kz: -2 * (np.cos(kx) + np.cos(ky) + np.cos(kz))
    >>> fig = plot_fermi_surface_3d(eps, mu=0.0, grid_size=20)
    >>> fig.data[0].type
    'isosurface'
    """
    grid = np.linspace(-k_range, k_range, grid_size)
    kx, ky, kz = np.meshgrid(grid, grid, grid, indexing="ij")
    values = dispersion_func(kx, ky, kz)
    fig = go.Figure(
        data=go.Isosurface(
            x=kx.flatten(),
            y=ky.flatten(),
            z=kz.flatten(),
            value=values.flatten(),
            isomin=mu,
            isomax=mu,
            surface_count=1,
            colorscale="Viridis",
            showscale=False,
        )
    )
    fig.update_layout(scene={"xaxis_title": "kx", "yaxis_title": "ky", "zaxis_title": "kz"})
    return fig
