r"""3D volumetric rendering of hydrogen electron density clouds (Plotly)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

from ..chapters.hydrogen_am import HydrogenOrbital, orbital_superposition_density

__all__ = ["orbital_density_grid", "plot_orbital_cloud", "animate_orbital_beating"]


def orbital_density_grid(orbital: HydrogenOrbital, extent: float | None = None, n_points: int = 60):
    r"""Evaluate :math:`\lvert\psi_{nlm}\rvert^2` on a Cartesian grid.

    Parameters
    ----------
    orbital : physicskit.quantum.chapters.hydrogen_am.HydrogenOrbital
        The orbital to evaluate.
    extent : float or None, optional
        Half-width of the cubic grid; defaults to a value scaled to the
        orbital's expected extent.
    n_points : int, default=60
        Number of grid points along each Cartesian axis.

    Returns
    -------
    X, Y, Z : numpy.ndarray
        Cartesian coordinate meshgrid.
    density : numpy.ndarray
        :math:`\lvert\psi_{nlm}\rvert^2` on the grid.
    """
    if extent is None:
        extent = 3 * orbital.n**2 * orbital.a0 / orbital.Z + 8

    lin = np.linspace(-extent, extent, n_points)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.where(r == 0, 1e-9, r)
    theta = np.arccos(np.clip(Z / r_safe, -1, 1))
    phi = np.arctan2(Y, X) % (2 * np.pi)

    density = orbital.density(r, theta, phi)
    return X, Y, Z, density


def plot_orbital_cloud(
    orbital: HydrogenOrbital, extent: float | None = None, n_points: int = 55, iso_fraction: float = 0.02, surface_count: int = 12
) -> go.Figure:
    r"""Volumetric isosurface render of the electron probability density.

    Renders :math:`\lvert\psi_{nlm}(r,\theta,\phi)\rvert^2` for a hydrogen
    orbital as a set of nested isosurfaces.

    Parameters
    ----------
    orbital : physicskit.quantum.chapters.hydrogen_am.HydrogenOrbital
        The orbital to render.
    extent : float or None, optional
        Half-width of the cubic grid; see :func:`orbital_density_grid`.
    n_points : int, default=55
        Number of grid points along each Cartesian axis.
    iso_fraction : float, default=0.02
        Lowest isosurface level, as a fraction of the peak density.
    surface_count : int, default=12
        Number of nested isosurfaces to draw.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    X, Y, Z, density = orbital_density_grid(orbital, extent, n_points)
    dmax = density.max()

    fig = go.Figure(
        data=go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=density.flatten(),
            isomin=iso_fraction * dmax,
            isomax=dmax,
            opacity=0.12,
            surface_count=surface_count,
            colorscale="Plasma",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )
    )
    fig.update_layout(
        title=f"|psi_{{{orbital.n},{orbital.l},{orbital.m}}}|^2",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    return fig


def animate_orbital_beating(
    orb_a: HydrogenOrbital,
    orb_b: HydrogenOrbital,
    times: np.ndarray,
    extent: float | None = None,
    n_points: int = 35,
    ca: complex = 2**-0.5,
    cb: complex = 2**-0.5,
    iso_fraction: float = 0.05,
    surface_count: int = 8,
) -> go.Figure:
    r"""Plotly ``frames``-based animation of a two-orbital density beating.

    Unlike a single stationary orbital (:func:`plot_orbital_cloud`, static in
    time since :math:`\lvert\psi_{nlm}\rvert^2` is time-independent), the
    density of a coherent superposition of two eigenstates
    (:func:`~physicskit.quantum.chapters.hydrogen_am.orbital_superposition_density`)
    genuinely reshapes at the Bohr frequency :math:`\omega_{ab}=E_a-E_b` --
    rendered here as a play-button-driven sequence of isosurface frames.

    Parameters
    ----------
    orb_a, orb_b : HydrogenOrbital
        The two eigenstates in the superposition.
    times : numpy.ndarray
        Times to render frames at.
    extent : float or None, optional
        Half-width of the cubic grid; defaults to a value scaled to
        ``orb_a``'s expected extent.
    n_points : int, default=35
        Number of grid points along each Cartesian axis (kept modest since
        a full 3D grid is evaluated at every frame).
    ca, cb : complex, default=1/sqrt(2) each
        Superposition amplitudes.
    iso_fraction : float, default=0.05
        Lowest isosurface level, as a fraction of the peak density (over all
        frames).
    surface_count : int, default=8
        Number of nested isosurfaces to draw per frame.

    Returns
    -------
    plotly.graph_objects.Figure
        With a "Play"/"Pause" button stepping through ``times``.
    """
    if extent is None:
        extent = 3 * orb_a.n**2 * orb_a.a0 / orb_a.Z + 8

    lin = np.linspace(-extent, extent, n_points)
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing="ij")
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.where(r == 0, 1e-9, r)
    theta = np.arccos(np.clip(Z / r_safe, -1, 1))
    phi = np.arctan2(Y, X) % (2 * np.pi)

    densities = [orbital_superposition_density(orb_a, orb_b, r, theta, phi, t, ca, cb) for t in times]
    dmax = max(d.max() for d in densities)

    def volume_trace(density):
        return go.Volume(
            x=X.flatten(),
            y=Y.flatten(),
            z=Z.flatten(),
            value=density.flatten(),
            isomin=iso_fraction * dmax,
            isomax=dmax,
            opacity=0.12,
            surface_count=surface_count,
            colorscale="Plasma",
            caps=dict(x_show=False, y_show=False, z_show=False),
        )

    frames = [go.Frame(data=[volume_trace(d)], name=f"{i}") for i, d in enumerate(densities)]

    fig = go.Figure(data=[volume_trace(densities[0])], frames=frames)
    fig.update_layout(
        title=f"|c_a psi_{{{orb_a.n},{orb_a.l},{orb_a.m}}} + c_b psi_{{{orb_b.n},{orb_b.l},{orb_b.m}}}|^2",
        scene=dict(xaxis_title="x", yaxis_title="y", zaxis_title="z", aspectmode="cube"),
        margin=dict(l=0, r=0, t=30, b=0),
        updatemenus=[
            dict(
                type="buttons",
                showactive=False,
                buttons=[
                    dict(label="Play", method="animate", args=[None, dict(frame=dict(duration=100, redraw=True), fromcurrent=True)]),
                    dict(label="Pause", method="animate", args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")]),
                ],
            )
        ],
    )
    return fig
